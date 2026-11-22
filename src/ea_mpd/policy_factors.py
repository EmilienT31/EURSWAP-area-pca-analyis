"""Window-consistent regressions using documented EA-MPD factor vintages.

Target belongs to the press-release window. Timing, Forward Guidance, and QE
belong jointly to the press-conference window. The factor CSVs are never
reconstructed from this project's PCA scores, rescaled, or imputed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from .data import EAMPDWorkbook, ValidationError, load_ea_mpd


PAPER_URL = "https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2281~3303fd281b.en.pdf"
APPENDIX_URL = "https://www.ecb.europa.eu/pub/pdf/annex/Appendix_Measuring_Euro_Area_Monetary_Policy.pdf"
WORKBOOK_URL = "https://www.ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx"
REQUIRED_FACTORS = ("target", "timing", "forward_guidance", "qe")
CONFERENCE_FACTORS = ("timing", "forward_guidance", "qe")
DISPLAY_NAMES = {
    "target": "Target",
    "timing": "Timing",
    "forward_guidance": "Forward Guidance",
    "qe": "QE",
}
EXPECTED_WINDOWS = {
    "target": "Press Release Window",
    "timing": "Press Conference Window",
    "forward_guidance": "Press Conference Window",
    "qe": "Press Conference Window",
}
PUBLISHED_NORMALIZATIONS = {
    "target": "unit effect on the 1-month OIS rate",
    "timing": "unit effect on the 6-month OIS rate",
    "forward_guidance": "unit effect on the 2-year OIS rate",
    "qe": "unit effect on the 10-year OIS rate",
}
EQUALITY_MATURITIES = ("OIS1M", "OIS2Y", "OIS10Y")
FILE_PATTERN = re.compile(
    r"^press_(release|conference)_factors_(\d{4}-\d{2}-\d{2})\.csv$"
)


@dataclass(frozen=True)
class FactorDefinition:
    canonical_name: str
    source_column: str
    units: str
    normalization: str
    event_window: str


@dataclass(frozen=True)
class FactorDataset:
    data: pd.DataFrame
    date_column: str
    definitions: dict[str, FactorDefinition]
    metadata: dict[str, Any]
    inventory: pd.DataFrame


def _normalise(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_dates(series: pd.Series, label: str) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce", format="mixed")
    if dates.isna().any():
        raise ValidationError(f"{label}: missing or invalid event dates.")
    dates = dates.dt.normalize()
    if dates.duplicated().any():
        duplicates = dates.loc[dates.duplicated(False)].dt.strftime("%Y-%m-%d").tolist()
        raise ValidationError(f"{label}: duplicate event dates {duplicates}.")
    if not dates.is_monotonic_increasing:
        raise ValidationError(f"{label}: event dates are not increasing.")
    return dates


def workbook_factor_audit(workbook: EAMPDWorkbook) -> pd.DataFrame:
    """Confirm that factor scores are external to the EA-MPD workbook."""

    headers = {
        _normalise(column)
        for frame in workbook.sheets.values()
        for column in frame.columns
    }
    aliases = {
        "target": {"TARGET", "TARGETFACTOR", "TARGETSURPRISE"},
        "timing": {"TIMING", "TIMINGFACTOR", "TIMINGSURPRISE"},
        "forward_guidance": {
            "FORWARDGUIDANCE",
            "FORWARDGUIDANCEFACTOR",
            "PATHFACTOR",
            "FGFACTOR",
        },
        "qe": {"QE", "QEFACTOR", "QUANTITATIVEEASING", "QUANTITATIVEEASINGFACTOR"},
    }
    rows = []
    for factor in REQUIRED_FACTORS:
        matches = sorted(headers & aliases[factor])
        rows.append(
            {
                "factor": factor,
                "provided_in_workbook": bool(matches),
                "matched_headers": ";".join(matches),
                "event_window": EXPECTED_WINDOWS[factor],
                "normalization": PUBLISHED_NORMALIZATIONS[factor],
                "units": "basis-point-equivalent surprise units",
                "status": "found_requires_validation" if matches else "external_csv_used",
            }
        )
    return pd.DataFrame(rows)


def inventory_factor_vintages(directory: Path) -> pd.DataFrame:
    """Inventory all observed factor files without assuming a preferred vintage."""

    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        window, vintage = match.groups()
        frame = pd.read_csv(path)
        dates = _parse_dates(frame.iloc[:, 0], path.name)
        rows.append(
            {
                "vintage": vintage,
                "window": window,
                "filename": path.name,
                "columns": ";".join(map(str, frame.columns)),
                "n_rows": len(frame),
                "date_start": dates.min(),
                "date_end": dates.max(),
                "missing_values": int(frame.isna().sum().sum()),
                "sha256": _sha256(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No factor-vintage CSVs found in {directory}.")
    inventory = pd.DataFrame(rows).sort_values(["vintage", "window"], ignore_index=True)
    counts = inventory.groupby("vintage")["window"].nunique()
    inventory["common_release_conference_vintage"] = inventory["vintage"].map(counts).eq(2)
    return inventory


def load_factor_vintage(directory: Path, vintage: str | None = None) -> FactorDataset:
    """Load one documented common release/conference factor vintage."""

    inventory = inventory_factor_vintages(directory)
    common = sorted(
        inventory.loc[inventory["common_release_conference_vintage"], "vintage"].unique()
    )
    if not common:
        raise ValidationError("No common press-release/press-conference vintage exists.")
    chosen = common[-1] if vintage is None else vintage
    if chosen not in common:
        raise ValidationError(
            f"Requested vintage {chosen!r} is not common to both windows; available: {common}."
        )

    release_path = directory / f"press_release_factors_{chosen}.csv"
    conference_path = directory / f"press_conference_factors_{chosen}.csv"
    release = pd.read_csv(release_path)
    conference = pd.read_csv(conference_path)
    if list(release.columns) != ["date", "target"]:
        raise ValidationError(
            f"Unexpected release schema {list(release.columns)}; observed schema is date,target."
        )
    if list(conference.columns) != ["date", "timing", "fg", "qe"]:
        raise ValidationError(
            "Unexpected conference schema "
            f"{list(conference.columns)}; observed schema is date,timing,fg,qe."
        )
    release["date"] = _parse_dates(release["date"], release_path.name)
    conference["date"] = _parse_dates(conference["date"], conference_path.name)
    for frame, names, label in [
        (release, ["target"], release_path.name),
        (conference, ["timing", "fg", "qe"], conference_path.name),
    ]:
        for name in names:
            numeric = pd.to_numeric(frame[name], errors="coerce")
            invalid = frame[name].notna() & numeric.isna()
            if invalid.any():
                raise ValidationError(f"{label}/{name}: non-numeric factor observations.")
            frame[name] = numeric

    factors = release.merge(conference, on="date", how="outer", validate="one_to_one")
    factors = factors.rename(columns={"fg": "forward_guidance"}).sort_values("date")
    definitions = {
        factor: FactorDefinition(
            factor,
            "fg" if factor == "forward_guidance" else factor,
            "basis-point-equivalent surprise units",
            PUBLISHED_NORMALIZATIONS[factor],
            EXPECTED_WINDOWS[factor],
        )
        for factor in REQUIRED_FACTORS
    }
    metadata = {
        "source": (
            "User-supplied EA-MPD factor CSVs; identification and normalization "
            "documented in Altavilla et al. (2019) and its factor-rotation appendix"
        ),
        "source_directory": str(directory.resolve()),
        "vintage": chosen,
        "release_file": release_path.name,
        "conference_file": conference_path.name,
        "release_sha256": _sha256(release_path),
        "conference_sha256": _sha256(conference_path),
        "paper_url": PAPER_URL,
        "appendix_url": APPENDIX_URL,
        "workbook_url": WORKBOOK_URL,
        "positive_sign": "restrictive monetary-policy surprise",
        "embedded_csv_metadata": False,
    }
    return FactorDataset(factors, "date", definitions, metadata, inventory)


def _find_window(workbook: EAMPDWorkbook, terms: tuple[str, ...]) -> str:
    matches = [
        name for name in workbook.sheets if all(term in name.casefold() for term in terms)
    ]
    if len(matches) != 1:
        raise ValidationError(f"Expected one event window matching {terms}, found {matches}.")
    return matches[0]


def _wide_ois(workbook: EAMPDWorkbook, window: str, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    frame = workbook.sheets[window]
    schema = workbook.schemas[window]
    canonical = [column.canonical_name for column in schema.ois_columns]
    rename = {column.source_name: f"{prefix}__{column.canonical_name}" for column in schema.ois_columns}
    result = frame[[schema.date_column, *rename]].rename(
        columns={schema.date_column: "event_date", **rename}
    )
    return result, canonical


def merge_factor_ois_events(
    workbook: EAMPDWorkbook, factors: FactorDataset
) -> tuple[pd.DataFrame, list[str], str, str]:
    """Merge without dropping missing factor or OIS observations."""

    release_window = _find_window(workbook, ("press", "release"))
    conference_window = _find_window(workbook, ("press", "conference"))
    release, maturities = _wide_ois(workbook, release_window, "release")
    conference, conference_maturities = _wide_ois(
        workbook, conference_window, "conference"
    )
    if maturities != conference_maturities:
        raise ValidationError("Release and conference OIS maturity grids differ.")

    factor_values = factors.data.rename(columns={factors.date_column: "event_date"})
    factor_dates = set(factor_values["event_date"])
    workbook_dates = set(release["event_date"]) | set(conference["event_date"])
    absent = sorted(factor_dates - workbook_dates)
    if absent:
        raise ValidationError(
            "Factor dates absent from the validated workbook: "
            f"{[date.date().isoformat() for date in absent]}."
        )

    merged = release.merge(conference, on="event_date", how="outer", validate="one_to_one")
    merged = merged.merge(factor_values, on="event_date", how="left", validate="one_to_one")
    release_ois = [f"release__{name}" for name in maturities]
    conference_ois = [f"conference__{name}" for name in maturities]
    merged["release_factor_available"] = merged["target"].notna()
    merged["conference_factors_complete"] = merged[list(CONFERENCE_FACTORS)].notna().all(axis=1)
    merged["release_ois_complete_16"] = merged[release_ois].notna().all(axis=1)
    merged["conference_ois_complete_16"] = merged[conference_ois].notna().all(axis=1)
    merged["release_common_sample"] = (
        merged["release_factor_available"] & merged["release_ois_complete_16"]
    )
    merged["conference_common_sample"] = (
        merged["conference_factors_complete"] & merged["conference_ois_complete_16"]
    )
    merged.insert(1, "factor_vintage", factors.metadata["vintage"])
    return merged.sort_values("event_date").reset_index(drop=True), maturities, release_window, conference_window


def _fit(y: pd.Series, x: pd.DataFrame):
    design = sm.add_constant(x, has_constant="add")
    return sm.OLS(y.astype(float), design.astype(float)).fit(cov_type="HC3")


def fit_maturity_regressions(
    merged: pd.DataFrame,
    maturities: list[str],
    factors: FactorDataset,
    release_window: str,
    conference_window: str,
) -> pd.DataFrame:
    """Estimate window-specific loading regressions on common maturity samples."""

    release_sample = merged.loc[merged["release_common_sample"]]
    conference_sample = merged.loc[merged["conference_common_sample"]]
    specs = {
        "target": (release_sample, "release", ["target"], release_window),
        "timing": (conference_sample, "conference", list(CONFERENCE_FACTORS), conference_window),
        "forward_guidance": (
            conference_sample,
            "conference",
            list(CONFERENCE_FACTORS),
            conference_window,
        ),
        "qe": (conference_sample, "conference", list(CONFERENCE_FACTORS), conference_window),
    }
    rows: list[dict[str, Any]] = []
    for factor, (sample, prefix, regressors, window) in specs.items():
        for maturity in maturities:
            result = _fit(sample[f"{prefix}__{maturity}"], sample[regressors])
            definition = factors.definitions[factor]
            rows.append(
                {
                    "factor": factor,
                    "maturity": maturity.removeprefix("OIS"),
                    "canonical_name": maturity,
                    "coefficient": result.params[factor],
                    "standard_error_hc3": result.bse[factor],
                    "p_value_hc3": result.pvalues[factor],
                    "ci95_lower": result.conf_int().loc[factor, 0],
                    "ci95_upper": result.conf_int().loc[factor, 1],
                    "n_events": int(result.nobs),
                    "sample_start": sample["event_date"].min(),
                    "sample_end": sample["event_date"].max(),
                    "r_squared": result.rsquared,
                    "dependent_variable_units": "basis points",
                    "factor_units": definition.units,
                    "normalization": definition.normalization,
                    "event_window": window,
                    "regressors": ";".join(regressors),
                    "factor_vintage": factors.metadata["vintage"],
                }
            )
    return pd.DataFrame(rows)


def equality_tests(
    merged: pd.DataFrame,
    factors: FactorDataset,
    release_window: str,
    conference_window: str,
) -> pd.DataFrame:
    """Test loading equality using event-level OIS differences and HC3 SEs."""

    pairs = [
        (EQUALITY_MATURITIES[0], EQUALITY_MATURITIES[1]),
        (EQUALITY_MATURITIES[1], EQUALITY_MATURITIES[2]),
        (EQUALITY_MATURITIES[0], EQUALITY_MATURITIES[2]),
    ]
    release_sample = merged.loc[merged["release_common_sample"]]
    conference_sample = merged.loc[merged["conference_common_sample"]]
    specs = {
        "target": (release_sample, "release", ["target"], release_window),
        "timing": (conference_sample, "conference", list(CONFERENCE_FACTORS), conference_window),
        "forward_guidance": (
            conference_sample,
            "conference",
            list(CONFERENCE_FACTORS),
            conference_window,
        ),
        "qe": (conference_sample, "conference", list(CONFERENCE_FACTORS), conference_window),
    }
    rows: list[dict[str, Any]] = []
    for factor, (sample, prefix, regressors, window) in specs.items():
        for left, right in pairs:
            difference = sample[f"{prefix}__{left}"] - sample[f"{prefix}__{right}"]
            result = _fit(difference, sample[regressors])
            rows.append(
                {
                    "factor": factor,
                    "shorter_maturity": left.removeprefix("OIS"),
                    "longer_maturity": right.removeprefix("OIS"),
                    "coefficient_difference": result.params[factor],
                    "standard_error_hc3": result.bse[factor],
                    "p_value_hc3": result.pvalues[factor],
                    "ci95_lower": result.conf_int().loc[factor, 0],
                    "ci95_upper": result.conf_int().loc[factor, 1],
                    "n_events": int(result.nobs),
                    "sample_start": sample["event_date"].min(),
                    "sample_end": sample["event_date"].max(),
                    "event_window": window,
                    "regressors": ";".join(regressors),
                    "test": "H0: beta_shorter = beta_longer",
                    "factor_vintage": factors.metadata["vintage"],
                }
            )
    return pd.DataFrame(rows)


def _plot_factor(coefficients: pd.DataFrame, factor: str, destination: Path) -> None:
    subset = coefficients.loc[coefficients["factor"].eq(factor)].reset_index(drop=True)
    positions = range(len(subset))
    errors = [
        subset["coefficient"] - subset["ci95_lower"],
        subset["ci95_upper"] - subset["coefficient"],
    ]
    fig, axis = plt.subplots(figsize=(9.5, 5.3))
    axis.errorbar(
        list(positions),
        subset["coefficient"],
        yerr=errors,
        fmt="o-",
        capsize=3,
        color="#125B8C",
        linewidth=1.8,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_xticks(list(positions), subset["maturity"])
    axis.set_xlabel("OIS maturity")
    axis.set_ylabel("OIS response (bp per factor unit)")
    title = DISPLAY_NAMES[factor]
    axis.set_title(f"{title} loading by maturity (HC3 95% CI)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _source_metadata(factors: FactorDataset, merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor, definition in factors.definitions.items():
        nonmissing = merged.loc[merged[factor].notna(), ["event_date", factor]]
        rows.append(
            {
                "factor": factor,
                "source_column": definition.source_column,
                "event_window": definition.event_window,
                "units": definition.units,
                "normalization": definition.normalization,
                "positive_sign": factors.metadata["positive_sign"],
                "factor_vintage": factors.metadata["vintage"],
                "n_nonmissing_events": len(nonmissing),
                "observed_start": nonmissing["event_date"].min(),
                "observed_end": nonmissing["event_date"].max(),
                "source_directory": factors.metadata["source_directory"],
                "release_file": factors.metadata["release_file"],
                "conference_file": factors.metadata["conference_file"],
                "release_sha256": factors.metadata["release_sha256"],
                "conference_sha256": factors.metadata["conference_sha256"],
                "embedded_csv_metadata": factors.metadata["embedded_csv_metadata"],
                "methodology_source": PAPER_URL,
                "rotation_source": APPENDIX_URL,
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    destination: Path,
    factors: FactorDataset,
    merged: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    release = merged.loc[merged["release_common_sample"]]
    conference = merged.loc[merged["conference_common_sample"]]
    qe_nonmissing = merged.loc[merged["qe"].notna(), "event_date"]
    strongest = (
        coefficients.assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coefficient")
        .groupby("factor", as_index=False)
        .tail(1)
        .sort_values("factor")
    )
    shape_lines = "\n".join(
        f"- **{DISPLAY_NAMES[row.factor]}** has its largest absolute loading at "
        f"{row.maturity}: {row.coefficient:.3f} bp per factor unit."
        for row in strongest.itertuples()
    )
    destination.write_text(
        f"""# Identified EA-MPD factor loading regressions

## Source and vintage

The selected common vintage is **{factors.metadata['vintage']}**, the latest date for which both an observed `press_release_factors` CSV and `press_conference_factors` CSV exist in the supplied directory. Historical factor estimates change across vintages, so release and conference factors are deliberately taken from the same vintage. Exact source hashes are recorded in `factor_source_metadata.csv`.

The CSVs contain no embedded citation or unit metadata. Identification, sign, and scaling are therefore documented from Altavilla et al. (2019): Target is a press-release factor; Timing, Forward Guidance, and QE are press-conference factors. Positive factor values are restrictive surprises. The factors are in basis-point-equivalent units, scaled to have unit effects at 1M, 6M, 2Y, and 10Y respectively.

## Event windows and samples

OIS dependent variables come from the validated EA-MPD workbook and remain in basis points. The workbook defines separate press-release and press-conference windows, including its timing change from July 2022; factors are never moved across windows.

- Target/release common sample: **{len(release)} events**, {release['event_date'].min().date()} to {release['event_date'].max().date()}.
- Timing/Forward Guidance/QE conference common sample: **{len(conference)} events**, {conference['event_date'].min().date()} to {conference['event_date'].max().date()}.
- QE is missing by construction before its observed factor sample; in this vintage its non-missing coverage is {qe_nonmissing.min().date()} to {qe_nonmissing.max().date()}.

The exported `merged_factor_ois_events.csv` preserves every workbook event and every missing value. The two common-sample flags show exactly which observations enter each model. No missing OIS change or factor is imputed.

## Regression design

At each of the 16 OIS maturities, the release-window change is regressed on Target. Separately, the conference-window change is regressed jointly on Timing, Forward Guidance, and QE. All models include an intercept and use HC3 heteroskedasticity-robust standard errors. Within each window every maturity uses the same events. Equality tests use event-level 1M-minus-2Y, 2Y-minus-10Y, and 1M-minus-10Y differences with the same regressors and HC3 inference.

The common-sample estimates at the normalization maturities are Target/1M {coefficients.query("factor == 'target' and maturity == '1M'")['coefficient'].item():.3f}, Timing/6M {coefficients.query("factor == 'timing' and maturity == '6M'")['coefficient'].item():.3f}, Forward Guidance/2Y {coefficients.query("factor == 'forward_guidance' and maturity == '2Y'")['coefficient'].item():.3f}, and QE/10Y {coefficients.query("factor == 'qe' and maturity == '10Y'")['coefficient'].item():.3f}. They need not equal exactly one because these regressions use later complete 16-maturity subsamples, whereas the source normalization was imposed in the factor-construction sample.

{shape_lines}

## Interpretation boundary

These regressions are descriptive factor-decomposition reconstructions. The factors were identified from risk-free rate changes that include the 1M, 3M, 6M, 1Y, 2Y, 5Y, and 10Y points used again as dependent variables. Normalization also anchors selected loadings by construction. Consequently, high fit and significant coefficients here are not independent causal identification. Causal applications should use outcomes not used to construct the factors or a separate design.

Primary methodology: [ECB Working Paper 2281]({PAPER_URL}) and its [factor-rotation appendix]({APPENDIX_URL}).
""",
        encoding="utf-8",
    )


def run(
    input_path: Path,
    factor_directory: Path,
    output_dir: Path,
    vintage: str | None = None,
) -> None:
    workbook = load_ea_mpd(input_path)
    factors = load_factor_vintage(factor_directory, vintage)
    merged, maturities, release_window, conference_window = merge_factor_ois_events(
        workbook, factors
    )
    coefficients = fit_maturity_regressions(
        merged, maturities, factors, release_window, conference_window
    )
    tests = equality_tests(merged, factors, release_window, conference_window)
    output_dir.mkdir(parents=True, exist_ok=True)

    workbook_factor_audit(workbook).to_csv(
        output_dir / "factor_source_audit.csv", index=False
    )
    factors.inventory.to_csv(
        output_dir / "factor_vintage_inventory.csv", index=False, date_format="%Y-%m-%d"
    )
    _source_metadata(factors, merged).to_csv(
        output_dir / "factor_source_metadata.csv", index=False, date_format="%Y-%m-%d"
    )
    merged.to_csv(
        output_dir / "merged_factor_ois_events.csv", index=False, date_format="%Y-%m-%d"
    )
    coefficients.to_csv(
        output_dir / "factor_coefficients_by_maturity.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    tests.to_csv(
        output_dir / "factor_effect_equality_tests.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    for factor in REQUIRED_FACTORS:
        _plot_factor(coefficients, factor, output_dir / f"{factor}_coefficients.png")
    _write_report(
        output_dir / "methodological_interpretation.md", factors, merged, coefficients
    )
    status = {
        "status": "completed",
        "factor_vintage": factors.metadata["vintage"],
        "release_common_events": int(merged["release_common_sample"].sum()),
        "conference_common_events": int(merged["conference_common_sample"].sum()),
        "pca_scores_relabelled": False,
        "missing_values_imputed": False,
        "swap_pricing_implemented": False,
    }
    (output_dir / "run_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run window-consistent EA-MPD factor loading regressions."
    )
    parser.add_argument("--input", type=Path, default=Path("Dataset_EA-MPD.xlsx"))
    parser.add_argument(
        "--factor-dir",
        type=Path,
        default=Path("data/external/ea_mpd_factors"),
    )
    parser.add_argument("--vintage", default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("results/policy_factor_regressions")
    )
    args = parser.parse_args()
    run(args.input, args.factor_dir, args.output, args.vintage)


if __name__ == "__main__":
    main()
