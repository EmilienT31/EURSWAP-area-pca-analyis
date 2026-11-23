import numpy as np
import pandas as pd
import pytest

from ea_mpd.pca_diagnostics import (
    BOOTSTRAP_PAIRS,
    INSTRUMENTS,
    METHODS,
    audit_error_metrics,
    select_bootstrap_comparisons,
    summarize_explained_variance,
)


def test_error_metric_audit_requires_common_128_event_panel() -> None:
    rows = [
        {
            "instrument": instrument,
            "approximation": method,
            "n_distinct_events": 128,
            "rmse_eur": 2.0,
            "mae_eur": 1.0,
            "normalized_rmse_bp_equivalent": 0.5,
        }
        for instrument in INSTRUMENTS
        for method in METHODS
    ]
    audited = audit_error_metrics(pd.DataFrame(rows))
    assert len(audited) == 20
    assert audited["n_distinct_events"].eq(128).all()

    broken = pd.DataFrame(rows)
    broken.loc[0, "n_distinct_events"] = 127
    with pytest.raises(ValueError, match="expected 4 x 5 x 128"):
        audit_error_metrics(broken)


def test_variance_summary_reports_event_distribution() -> None:
    rows = []
    for event in pd.date_range("2020-01-01", periods=128):
        for component, share in ((1, 0.9), (2, 0.08), (3, 0.02)):
            rows.append(
                {
                    "event_date": event,
                    "component": component,
                    "explained_variance_ratio": share,
                    "cumulative_explained_variance_ratio": sum(
                        value for _, value in ((1, 0.9), (2, 0.08), (3, 0.02))
                        if _ <= component
                    ),
                }
            )
    summary = summarize_explained_variance(pd.DataFrame(rows))
    assert summary.loc[summary["component"].eq(1), "explained_mean"].iloc[0] == pytest.approx(0.9)
    assert summary.loc[summary["component"].eq(3), "cumulative_mean"].iloc[0] == pytest.approx(1.0)


def test_selected_bootstrap_interprets_paired_mse_difference() -> None:
    rows = []
    for method_a, method_b in BOOTSTRAP_PAIRS:
        for instrument in (*INSTRUMENTS, "ALL"):
            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "instrument": instrument,
                    "n_events": 128,
                    "mean_squared_error_difference_eur2": 2.0,
                    "ci_2_5pct_eur2": 1.0,
                    "ci_97_5pct_eur2": 3.0,
                    "n_bootstrap": 100,
                    "seed": 1,
                    "interpretation": "negative values favor method_a",
                }
            )
    selected = select_bootstrap_comparisons(pd.DataFrame(rows))
    assert len(selected) == 20
    assert selected["ci_excludes_zero"].all()
    assert (selected["favored_method_95pct"] == selected["method_b"]).all()
