import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from ea_mpd.pca_pnl import (
    METHOD_COLUMNS,
    bootstrap_squared_error_differences,
    build_daily_curve_changes,
    fit_covariance_pca,
    trailing_training_sample,
)
from ea_mpd.swap_risk_experiments import NODE_GRID


def test_daily_changes_and_training_window_are_strictly_pre_event() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    rows = []
    for date_index, date in enumerate(dates):
        for maturity in NODE_GRID:
            rows.append(
                {
                    "date": date,
                    "maturity_years": maturity,
                    "zero_rate_decimal": 0.02 + 0.0001 * date_index * maturity,
                }
            )
    changes = build_daily_curve_changes(pd.DataFrame(rows))
    assert len(changes) == len(dates) - 1
    assert changes.iloc[0, 4] == pytest.approx(5.0)

    event_date = dates[6]
    sample = trailing_training_sample(changes, event_date, max_observations=3)
    assert len(sample) == 3
    assert sample.index.max() < event_date
    assert event_date not in sample.index


def test_covariance_pca_signs_units_and_full_reconstruction() -> None:
    rng = np.random.default_rng(1234)
    maturity = np.asarray(NODE_GRID)
    level = np.ones(10)
    slope = maturity - maturity.mean()
    curvature = np.exp(-0.5 * ((maturity - 5.5) / 1.5) ** 2) - 0.25
    scores = rng.normal(size=(600, 3)) * np.array([6.0, 2.0, 0.7])
    changes_bp = scores @ np.vstack([level, slope, curvature]) + rng.normal(
        scale=0.05, size=(600, 10)
    )
    model = fit_covariance_pca(changes_bp)

    assert model.loadings.mean(axis=0)[0] > 0
    assert np.corrcoef(model.loadings[:, 1], maturity)[0, 1] > 0
    belly = model.loadings[3:7, 2].mean()
    ends = model.loadings[[0, 1, 8, 9], 2].mean()
    assert belly - ends > 0
    np.testing.assert_allclose(model.loadings.T @ model.loadings, np.eye(10), atol=1e-12)
    assert model.explained_variance_ratio.sum() == pytest.approx(1.0)

    shock_bp = changes_bp[0] + 1.0
    reconstruction = model.reconstruct(shock_bp, 10)
    np.testing.assert_allclose(reconstruction, shock_bp, atol=1e-11)
    krd_eur_per_bp = np.linspace(-20.0, 100.0, 10)
    score = model.scores(shock_bp)
    contribution = float(krd_eur_per_bp @ model.mean_bp)
    contribution += sum(
        float(krd_eur_per_bp @ (model.loadings[:, index] * score[index]))
        for index in range(10)
    )
    assert contribution == pytest.approx(float(krd_eur_per_bp @ shock_bp), abs=1e-9)


def test_event_cluster_bootstrap_is_paired_and_reproducible() -> None:
    rows = []
    for event_index, event_date in enumerate(pd.date_range("2020-01-01", periods=6)):
        for instrument_index, instrument in enumerate(("2Y_payer", "10Y_payer")):
            actual = 100.0 * event_index + 10.0 * instrument_index
            row = {
                "event_date": event_date,
                "instrument": instrument,
                "full_repricing_pnl_eur": actual,
            }
            for method_index, column in enumerate(METHOD_COLUMNS.values(), start=1):
                row[column] = actual + method_index * (event_index - 2.0)
            rows.append(row)
    frame = pd.DataFrame(rows)
    first = bootstrap_squared_error_differences(frame, n_bootstrap=100, seed=42)
    second = bootstrap_squared_error_differences(frame, n_bootstrap=100, seed=42)
    pdt.assert_frame_equal(first, second)
    pooled = first.loc[first["instrument"].eq("ALL")]
    assert pooled["n_events"].eq(6).all()
    assert pooled["n_bootstrap"].eq(100).all()
