from ea_mpd.robustness import mean_absolute_sample_comparison

from test_analysis import _test_workbook


def test_common_sample_uses_identical_event_count_at_every_maturity() -> None:
    comparison = mean_absolute_sample_comparison(
        _test_workbook(), "Monetary Event Window"
    )
    common = comparison.loc[
        comparison["sample_definition"].eq("common_complete_16_maturities")
    ]

    assert common["n_events"].nunique() == 1
    assert common["n_events"].iloc[0] == 3
