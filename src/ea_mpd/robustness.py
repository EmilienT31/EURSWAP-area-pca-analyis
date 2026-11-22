"""Robustness analysis for sample composition and OIS-curve PCA."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .analysis import PCAResult, descriptive_statistics, extract_ois_surprises, run_pca
from .data import EAMPDWorkbook, find_monetary_event_sheet, load_ea_mpd


REDUCED_GRID = ("OIS1M", "OIS3M", "OIS6M", "OIS1Y", "OIS2Y", "OIS3Y")
FULL_SPEC = "full_16_maturities"
REDUCED_SPEC = "reduced_6_maturities"


def _complete_frame(
    workbook: EAMPDWorkbook,
    event_sheet: str,
    canonical_names: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list]:
    frame = workbook.sheets[event_sheet]
    schema = workbook.schemas[event_sheet]
    columns = list(schema.ois_columns)
    if canonical_names is not None:
        lookup = {column.canonical_name: column for column in columns}
        missing = [name for name in canonical_names if name not in lookup]
        if missing:
            raise ValueError(f"Requested maturities are absent: {missing}.")
        columns = [lookup[name] for name in canonical_names]
    source_names = [column.source_name for column in columns]
    complete = frame[[schema.date_column, *source_names]].dropna().copy()
    return complete, columns


def mean_absolute_sample_comparison(
    workbook: EAMPDWorkbook, event_sheet: str
) -> pd.DataFrame:
    """Compare per-maturity available samples with one complete 16-point sample."""

    surprises = extract_ois_surprises(workbook)
    all_stats = descriptive_statistics(surprises)
    all_available = all_stats.loc[all_stats["window"].eq(event_sheet)].copy()
    event_surprises = surprises.loc[surprises["window"].eq(event_sheet)].copy()
    ranges = (
        event_surprises.dropna(subset=["surprise_bp"])
        .groupby("canonical_name")["event_date"]
        .agg(sample_start="min", sample_end="max")
        .reset_index()
    )
    all_available = all_available.merge(ranges, on="canonical_name", how="left")
    all_available = all_available.rename(columns={"n_observations": "n_events"})
    all_available["sample_definition"] = "all_available_by_maturity"

    complete, columns = _complete_frame(workbook, event_sheet)
    schema = workbook.schemas[event_sheet]
    complete_long = complete.melt(
        id_vars=schema.date_column,
        value_vars=[column.source_name for column in columns],
        var_name="source_column",
        value_name="surprise_bp",
    )
    metadata = {column.source_name: column for column in columns}
    common = (
        complete_long.groupby("source_column")["surprise_bp"]
        .agg(n_events="count", mean_abs_bp=lambda values: values.abs().mean())
        .reset_index()
    )
    common["canonical_name"] = common["source_column"].map(
        lambda name: metadata[name].canonical_name
    )
    common["maturity"] = common["source_column"].map(
        lambda name: metadata[name].maturity_label
    )
    common["maturity_years"] = common["source_column"].map(
        lambda name: metadata[name].maturity_years
    )
    common["sample_start"] = complete[schema.date_column].min()
    common["sample_end"] = complete[schema.date_column].max()
    common["sample_definition"] = "common_complete_16_maturities"

    columns_to_keep = [
        "sample_definition",
        "canonical_name",
        "maturity",
        "maturity_years",
        "n_events",
        "sample_start",
        "sample_end",
        "mean_abs_bp",
    ]
    combined = pd.concat(
        [all_available[columns_to_keep], common[columns_to_keep]],
        ignore_index=True,
    )
    return combined.sort_values(
        ["sample_definition", "maturity_years"], ignore_index=True
    )


def _pca_with_spec(result: PCAResult, spec: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    explained = result.explained_variance.copy()
    explained.insert(0, "specification", spec)
    explained.insert(1, "n_events", result.n_complete_events)
    explained.insert(2, "sample_start", result.scores["event_date"].min())
    explained.insert(3, "sample_end", result.scores["event_date"].max())
    explained.insert(4, "n_maturities", len(result.loadings))
    loadings = result.loadings.copy()
    loadings.insert(0, "specification", spec)
    return explained, loadings


def _plot_mean_absolute(comparison: pd.DataFrame, destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5.5))
    styles = {
        "all_available_by_maturity": ("Available observations at each maturity", "#7A7A7A", "o"),
        "common_complete_16_maturities": ("Common complete 16-maturity sample", "#125B8C", "s"),
    }
    ordered_labels = (
        comparison[["maturity", "maturity_years"]]
        .drop_duplicates()
        .sort_values("maturity_years")
    )
    positions = range(len(ordered_labels))
    for definition, (label, colour, marker) in styles.items():
        subset = comparison.loc[comparison["sample_definition"].eq(definition)].sort_values(
            "maturity_years"
        )
        axis.plot(
            list(positions),
            subset["mean_abs_bp"],
            label=label,
            color=colour,
            marker=marker,
            linewidth=2,
        )
    axis.set_xticks(list(positions), ordered_labels["maturity"])
    axis.set_xlabel("OIS maturity (different samples only for grey line)")
    axis.set_ylabel("Mean absolute surprise (basis points)")
    axis.set_title("OIS surprise magnitude: unequal versus common event sample")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_loadings(
    full_loadings: pd.DataFrame,
    reduced_loadings: pd.DataFrame,
    destination: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
    palette = {"PC1": "#125B8C", "PC2": "#C55A11", "PC3": "#4F7F3A"}
    panels = [
        (axes[0], full_loadings, "16-maturity grid"),
        (axes[1], reduced_loadings, "Reduced 6-maturity grid"),
    ]
    for axis, loadings, title in panels:
        positions = range(len(loadings))
        for component, colour in palette.items():
            axis.plot(
                list(positions),
                loadings[component],
                marker="o",
                linewidth=2,
                label=component,
                color=colour,
            )
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xticks(list(positions), loadings["maturity"])
        axis.set_xlabel("OIS maturity")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("PCA loading")
    axes[1].legend(frameon=False)
    fig.suptitle("Event-window OIS PCA loadings (centered, unstandardized)")
    fig.text(
        0.5,
        0.01,
        "Panels use different maturity grids and different complete-case event samples; shapes are not point-for-point comparable.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _factor_extrema(loadings: pd.DataFrame, component: str) -> str:
    minimum = loadings.loc[loadings[component].idxmin()]
    maximum = loadings.loc[loadings[component].idxmax()]
    return (
        f"minimum at {minimum['maturity']} ({minimum[component]:.3f}) and "
        f"maximum at {maximum['maturity']} ({maximum[component]:.3f})"
    )


def _write_report(
    destination: Path,
    comparison: pd.DataFrame,
    full: PCAResult,
    reduced: PCAResult,
) -> None:
    common = comparison.loc[
        comparison["sample_definition"].eq("common_complete_16_maturities")
    ]
    unequal = comparison.loc[
        comparison["sample_definition"].eq("all_available_by_maturity")
    ]
    joined = unequal.merge(
        common,
        on=["canonical_name", "maturity", "maturity_years"],
        suffixes=("_unequal", "_common"),
    )
    joined["difference_bp"] = joined["mean_abs_bp_common"] - joined["mean_abs_bp_unequal"]
    largest_difference = joined.loc[joined["difference_bp"].abs().idxmax()]

    full_three = full.explained_variance.head(3)
    reduced_three = reduced.explained_variance.head(3)
    full_start, full_end = full.scores["event_date"].min(), full.scores["event_date"].max()
    reduced_start, reduced_end = (
        reduced.scores["event_date"].min(),
        reduced.scores["event_date"].max(),
    )
    report = f"""# Robustness analysis: common samples and PCA maturity grids

All OIS changes are in basis points. Both PCAs center each maturity and use the covariance matrix without variance standardization. Missing surprises are never imputed.

## Mean absolute surprises

The unequal-sample line uses every available observation separately at each maturity, so its maturity-specific sample size ranges from **{int(unequal['n_events'].min())} to {int(unequal['n_events'].max())}** events. The common-sample line uses the same **{int(common['n_events'].iloc[0])}** complete events at all 16 maturities, from **{pd.Timestamp(common['sample_start'].iloc[0]).date()} to {pd.Timestamp(common['sample_end'].iloc[0]).date()}**.

The largest common-minus-unequal change occurs at **{largest_difference['maturity']}**: {largest_difference['difference_bp']:+.2f} bp. Differences between the lines combine genuine time variation in announcement volatility with the changing historical coverage of longer OIS maturities; they should not be read as a mechanical maturity effect.

## PCA samples and explained variance

- **Full 16-maturity specification:** {full.n_complete_events} events, {full_start.date()} to {full_end.date()}. PC1/PC2/PC3 explain {full_three.iloc[0]['explained_variance_ratio']:.1%}, {full_three.iloc[1]['explained_variance_ratio']:.1%}, and {full_three.iloc[2]['explained_variance_ratio']:.1%}; cumulatively {full_three.iloc[2]['cumulative_explained_variance_ratio']:.1%}.
- **Reduced six-maturity specification (1M–3Y):** {reduced.n_complete_events} events, {reduced_start.date()} to {reduced_end.date()}. PC1/PC2/PC3 explain {reduced_three.iloc[0]['explained_variance_ratio']:.1%}, {reduced_three.iloc[1]['explained_variance_ratio']:.1%}, and {reduced_three.iloc[2]['explained_variance_ratio']:.1%}; cumulatively {reduced_three.iloc[2]['cumulative_explained_variance_ratio']:.1%}.

## Factor shapes

For the full grid, PC1 is { _factor_extrema(full.loadings, 'PC1') }; PC2 is { _factor_extrema(full.loadings, 'PC2') }; PC3 is { _factor_extrema(full.loadings, 'PC3') }. PC1 is a broad same-sign path factor with its strongest loading in the medium maturities, PC2 rotates short against long rates, and PC3 contrasts the belly with the ends.

For the reduced grid, PC1 is { _factor_extrema(reduced.loadings, 'PC1') }; PC2 is { _factor_extrema(reduced.loadings, 'PC2') }; PC3 is { _factor_extrema(reduced.loadings, 'PC3') }. Removing the 4Y–20Y points changes both the covariance matrix and the events admitted to estimation, so changes in loadings cannot be attributed to maturity-grid truncation alone. The second panel therefore shows a robustness specification, not a direct continuation of the first curve.

Component signs are normalized deterministically for reproducibility, but PCA signs have no intrinsic economic meaning. Interpretation should focus on relative loading shapes and co-movement.
"""
    destination.write_text(report, encoding="utf-8")


def run(input_path: Path, output_dir: Path) -> None:
    workbook = load_ea_mpd(input_path)
    event_sheet = find_monetary_event_sheet(workbook)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = mean_absolute_sample_comparison(workbook, event_sheet)
    full = run_pca(workbook, event_sheet)
    reduced = run_pca(workbook, event_sheet, REDUCED_GRID)
    full_explained, full_loadings = _pca_with_spec(full, FULL_SPEC)
    reduced_explained, reduced_loadings = _pca_with_spec(reduced, REDUCED_SPEC)

    comparison.to_csv(
        output_dir / "mean_abs_surprise_sample_comparison.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    pd.concat([full_explained, reduced_explained], ignore_index=True).to_csv(
        output_dir / "pca_explained_variance_comparison.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    pd.concat([full_loadings, reduced_loadings], ignore_index=True).to_csv(
        output_dir / "pca_loadings_comparison.csv", index=False
    )
    full.scores.to_csv(
        output_dir / "pca_scores_full16.csv", index=False, date_format="%Y-%m-%d"
    )
    reduced.scores.to_csv(
        output_dir / "pca_scores_reduced6.csv", index=False, date_format="%Y-%m-%d"
    )

    sample_summary = pd.DataFrame(
        [
            {
                "analysis": "mean_absolute_surprise",
                "specification": "all_available_by_maturity",
                "maturity_grid": "1W,1M,3M,6M,1Y,2Y,3Y,4Y,5Y,6Y,7Y,8Y,9Y,10Y,15Y,20Y",
                "n_maturities": 16,
                "n_events_min": int(comparison.loc[comparison["sample_definition"].eq("all_available_by_maturity"), "n_events"].min()),
                "n_events_max": int(comparison.loc[comparison["sample_definition"].eq("all_available_by_maturity"), "n_events"].max()),
                "sample_start": comparison.loc[comparison["sample_definition"].eq("all_available_by_maturity"), "sample_start"].min(),
                "sample_end": comparison.loc[comparison["sample_definition"].eq("all_available_by_maturity"), "sample_end"].max(),
            },
            {
                "analysis": "mean_absolute_surprise",
                "specification": "common_complete_16_maturities",
                "maturity_grid": "1W,1M,3M,6M,1Y,2Y,3Y,4Y,5Y,6Y,7Y,8Y,9Y,10Y,15Y,20Y",
                "n_maturities": 16,
                "n_events_min": full.n_complete_events,
                "n_events_max": full.n_complete_events,
                "sample_start": full.scores["event_date"].min(),
                "sample_end": full.scores["event_date"].max(),
            },
            {
                "analysis": "pca",
                "specification": FULL_SPEC,
                "maturity_grid": "1W,1M,3M,6M,1Y,2Y,3Y,4Y,5Y,6Y,7Y,8Y,9Y,10Y,15Y,20Y",
                "n_maturities": 16,
                "n_events_min": full.n_complete_events,
                "n_events_max": full.n_complete_events,
                "sample_start": full.scores["event_date"].min(),
                "sample_end": full.scores["event_date"].max(),
            },
            {
                "analysis": "pca",
                "specification": REDUCED_SPEC,
                "maturity_grid": "1M,3M,6M,1Y,2Y,3Y",
                "n_maturities": 6,
                "n_events_min": reduced.n_complete_events,
                "n_events_max": reduced.n_complete_events,
                "sample_start": reduced.scores["event_date"].min(),
                "sample_end": reduced.scores["event_date"].max(),
            },
        ]
    )
    sample_summary.to_csv(
        output_dir / "sample_summary.csv", index=False, date_format="%Y-%m-%d"
    )
    _plot_mean_absolute(comparison, output_dir / "mean_abs_surprise_comparison.png")
    _plot_loadings(
        full.loadings,
        reduced.loadings,
        output_dir / "pca_loadings_comparison.png",
    )
    _write_report(output_dir / "robustness_report.md", comparison, full, reduced)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run common-sample and PCA-grid robustness checks."
    )
    parser.add_argument("--input", type=Path, default=Path("Dataset_EA-MPD.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("results/robustness"))
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()

