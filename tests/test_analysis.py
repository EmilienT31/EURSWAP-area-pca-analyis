from pathlib import Path

import pandas as pd

from ea_mpd.analysis import descriptive_statistics, extract_ois_surprises, run_pca
from ea_mpd.data import EAMPDWorkbook, OISColumn, SheetSchema


def _test_workbook() -> EAMPDWorkbook:
    """Small deterministic unit-test data, never presented as research output."""

    frame = pd.DataFrame(
        {
            "event": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "short": [1.0, 0.0, -1.0],
            "long": [2.0, 0.0, -2.0],
        }
    )
    columns = (
        OISColumn("short", "OIS1M", "1M", 1 / 12),
        OISColumn("long", "OIS1Y", "1Y", 1.0),
    )
    return EAMPDWorkbook(
        Path("unit-test-only.xlsx"),
        (),
        {"Monetary Event Window": frame},
        {"Monetary Event Window": SheetSchema("event", columns)},
    )


def test_extraction_and_statistics_preserve_basis_point_changes() -> None:
    surprises = extract_ois_surprises(_test_workbook())
    stats = descriptive_statistics(surprises)

    assert len(surprises) == 6
    assert stats.loc[stats["maturity"] == "1Y", "mean_abs_bp"].item() == 4 / 3


def test_pca_reports_complete_sample_and_dominant_first_component() -> None:
    result = run_pca(_test_workbook(), "Monetary Event Window")

    assert result.n_total_events == 3
    assert result.n_complete_events == 3
    assert result.explained_variance.loc[0, "explained_variance_ratio"] > 0.999


def test_reduced_maturity_pca_can_use_a_longer_complete_sample() -> None:
    frame = pd.DataFrame(
        {
            "event": pd.date_range("2020-01-01", periods=5, freq="MS"),
            "one_month": [1.0, -1.0, 2.0, -2.0, 0.5],
            "one_year": [2.0, -2.0, 1.0, -1.0, 0.0],
            "long": [3.0, -3.0, 2.0, -2.0, None],
        }
    )
    columns = (
        OISColumn("one_month", "OIS1M", "1M", 1 / 12),
        OISColumn("one_year", "OIS1Y", "1Y", 1.0),
        OISColumn("long", "OIS10Y", "10Y", 10.0),
    )
    workbook = EAMPDWorkbook(
        Path("unit-test-only.xlsx"),
        (),
        {"Monetary Event Window": frame},
        {"Monetary Event Window": SheetSchema("event", columns)},
    )

    full = run_pca(workbook, "Monetary Event Window")
    reduced = run_pca(
        workbook, "Monetary Event Window", ("OIS1M", "OIS1Y")
    )

    assert full.n_complete_events == 4
    assert reduced.n_complete_events == 5
    assert reduced.loadings["maturity"].tolist() == ["1M", "1Y"]

