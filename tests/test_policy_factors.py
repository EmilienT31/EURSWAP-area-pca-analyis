from pathlib import Path

import pandas as pd

from ea_mpd.data import EAMPDWorkbook, OISColumn, SheetSchema
from ea_mpd.policy_factors import (
    FactorDataset,
    FactorDefinition,
    equality_tests,
    fit_maturity_regressions,
    load_factor_vintage,
    merge_factor_ois_events,
    workbook_factor_audit,
)


def _workbook() -> EAMPDWorkbook:
    """Deterministic unit-test fixture, never used as research output."""

    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    target = pd.Series(range(-6, 6), dtype=float)
    timing = pd.Series([(i % 4) - 1.5 for i in range(12)], dtype=float)
    fg = pd.Series([(-1) ** i * (i + 1) for i in range(12)], dtype=float)
    qe = pd.Series([(i % 3) - 1 for i in range(12)], dtype=float)
    release = pd.DataFrame(
        {
            "date": dates,
            "one_month": 1.0 * target,
            "two_year": 0.5 * target,
            "ten_year": 0.1 * target,
        }
    )
    conference = pd.DataFrame(
        {
            "date": dates,
            "one_month": 0.1 * timing + 0.2 * fg + 0.1 * qe,
            "two_year": 0.8 * timing + 1.0 * fg + 0.4 * qe,
            "ten_year": 0.3 * timing + 0.3 * fg + 1.0 * qe,
        }
    )
    columns = (
        OISColumn("one_month", "OIS1M", "1M", 1 / 12),
        OISColumn("two_year", "OIS2Y", "2Y", 2.0),
        OISColumn("ten_year", "OIS10Y", "10Y", 10.0),
    )
    return EAMPDWorkbook(
        Path("test-only.xlsx"),
        (),
        {"Press Release Window": release, "Press Conference Window": conference},
        {
            "Press Release Window": SheetSchema("date", columns),
            "Press Conference Window": SheetSchema("date", columns),
        },
    )


def _factors(qe_missing_first: bool = False) -> FactorDataset:
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    qe = pd.Series([(i % 3) - 1 for i in range(12)], dtype=float)
    if qe_missing_first:
        qe.iloc[0] = None
    data = pd.DataFrame(
        {
            "date": dates,
            "target": range(-6, 6),
            "timing": [(i % 4) - 1.5 for i in range(12)],
            "forward_guidance": [(-1) ** i * (i + 1) for i in range(12)],
            "qe": qe,
        }
    )
    definitions = {
        "target": FactorDefinition("target", "target", "bp", "1M unit", "Press Release Window"),
        "timing": FactorDefinition("timing", "timing", "bp", "6M unit", "Press Conference Window"),
        "forward_guidance": FactorDefinition(
            "forward_guidance", "fg", "bp", "2Y unit", "Press Conference Window"
        ),
        "qe": FactorDefinition("qe", "qe", "bp", "10Y unit", "Press Conference Window"),
    }
    return FactorDataset(
        data,
        "date",
        definitions,
        {"vintage": "test", "positive_sign": "restrictive"},
        pd.DataFrame(),
    )


def test_workbook_audit_does_not_relabel_ois_or_pca_columns() -> None:
    audit = workbook_factor_audit(_workbook())
    assert not audit["provided_in_workbook"].any()
    assert set(audit["status"]) == {"external_csv_used"}
    assert set(audit["factor"]) == {"target", "timing", "forward_guidance", "qe"}


def test_missing_factor_is_preserved_and_excluded_only_from_conference_sample() -> None:
    merged, _, _, _ = merge_factor_ois_events(_workbook(), _factors(qe_missing_first=True))

    assert merged["qe"].isna().sum() == 1
    assert merged["release_common_sample"].sum() == 12
    assert merged["conference_common_sample"].sum() == 11


def test_common_sample_regressions_recover_window_specific_loadings() -> None:
    factors = _factors()
    merged, maturities, release_window, conference_window = merge_factor_ois_events(
        _workbook(), factors
    )
    coefficients = fit_maturity_regressions(
        merged, maturities, factors, release_window, conference_window
    )

    expected = [
        ("target", "OIS1M", 1.0),
        ("timing", "OIS2Y", 0.8),
        ("forward_guidance", "OIS2Y", 1.0),
        ("qe", "OIS10Y", 1.0),
    ]
    for factor, maturity, value in expected:
        actual = coefficients.query(
            "factor == @factor and canonical_name == @maturity"
        )["coefficient"].item()
        assert abs(actual - value) < 1e-10
    assert coefficients.query("factor == 'target'")["n_events"].unique().tolist() == [12]
    assert coefficients.query("factor != 'target'")["n_events"].unique().tolist() == [12]

    tests = equality_tests(merged, factors, release_window, conference_window)
    target_short_long = tests.query(
        "factor == 'target' and shorter_maturity == '1M' and longer_maturity == '10Y'"
    )
    assert abs(target_short_long["coefficient_difference"].item() - 0.9) < 1e-10


def test_loader_chooses_latest_common_vintage_and_keeps_qe_missing(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="MS")
    for vintage in ["2020-02-01", "2020-03-01"]:
        pd.DataFrame({"date": dates, "target": [1.0, 2.0, 3.0]}).to_csv(
            tmp_path / f"press_release_factors_{vintage}.csv", index=False
        )
        pd.DataFrame(
            {
                "date": dates,
                "timing": [1.0, 2.0, 3.0],
                "fg": [2.0, 3.0, 4.0],
                "qe": [None, 1.0, 2.0],
            }
        ).to_csv(tmp_path / f"press_conference_factors_{vintage}.csv", index=False)

    factors = load_factor_vintage(tmp_path)

    assert factors.metadata["vintage"] == "2020-03-01"
    assert factors.data["qe"].isna().sum() == 1
    assert "forward_guidance" in factors.data
