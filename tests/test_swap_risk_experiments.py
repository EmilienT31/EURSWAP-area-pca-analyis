import numpy as np
import pandas as pd
import pytest

from ea_mpd.ecb_curve_data import SPOT_TYPES
from ea_mpd.swap_pricing import (
    OISSwap,
    YieldCurve,
    bootstrap_par_swap_curve,
    key_rate_sensitivities,
    parallel_dv01,
)
from ea_mpd.swap_risk_experiments import (
    EventCurveState,
    NODE_GRID,
    build_event_eligibility,
    dv01_neutral_curve_portfolio,
    zero_risk_decomposition,
)


def _state() -> EventCurveState:
    terms = np.asarray(NODE_GRID)
    base = YieldCurve(terms, np.linspace(0.02, 0.03, len(terms)))
    base_par = np.array([base.par_swap_rate(term) for term in terms])
    par_shocks_bp = np.array([-2.0, -1.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    shocked_par = base_par + par_shocks_bp * 1.0e-4
    shocked = bootstrap_par_swap_curve(terms, shocked_par)
    zero_shocks = (shocked.zero_rates_cc - base.zero_rates_cc) * 10_000
    return EventCurveState(
        pd.Timestamp("2025-01-30"),
        base,
        shocked,
        base_par,
        shocked_par,
        par_shocks_bp,
        zero_shocks,
        float(zero_shocks.mean()),
    )


def test_parallel_and_shape_decomposition_reconstructs_krd() -> None:
    state = _state()
    swap = OISSwap.at_par(state.base_curve, 10_000_000, 10.0)
    krd = key_rate_sensitivities(swap, state.base_curve, NODE_GRID)[
        "sensitivity_eur_per_bp"
    ]
    result = zero_risk_decomposition(
        krd, parallel_dv01(swap, state.base_curve), state.zero_shocks_bp
    )
    assert result["zero_krd_approximation_eur"] == pytest.approx(
        result["parallel_only_approximation_eur"]
        + result["residual_shape_krd_approximation_eur"],
        abs=0.25,
    )


def test_curve_portfolio_cancels_parallel_but_retains_key_risk() -> None:
    state = _state()
    payer, receiver = dv01_neutral_curve_portfolio(state)
    portfolio_parallel = parallel_dv01(payer, state.base_curve) + parallel_dv01(
        receiver, state.base_curve
    )
    payer_krd = key_rate_sensitivities(payer, state.base_curve, NODE_GRID)
    receiver_krd = key_rate_sensitivities(receiver, state.base_curve, NODE_GRID)
    portfolio_krd = (
        payer_krd["sensitivity_eur_per_bp"]
        + receiver_krd["sensitivity_eur_per_bp"]
    )
    assert portfolio_parallel == pytest.approx(0.0, abs=1.0e-8)
    assert portfolio_krd.abs().max() > 100.0


def test_eligibility_uses_exact_dates_and_preserves_missing_shocks() -> None:
    exact_date = pd.Timestamp("2025-01-30")
    missing_date = pd.Timestamp("2025-03-06")
    curve_history = pd.DataFrame(
        {
            "date": [exact_date] * len(SPOT_TYPES),
            "maturity_years": np.linspace(0.25, 10.0, len(SPOT_TYPES)),
            "zero_rate_decimal": np.linspace(0.02, 0.03, len(SPOT_TYPES)),
        }
    )
    rows = []
    for event_date in (exact_date, missing_date):
        for maturity in NODE_GRID:
            rows.append(
                {
                    "event_date": event_date,
                    "maturity_years": maturity,
                    "surprise_bp": (
                        np.nan if event_date == missing_date and maturity == 10.0 else 1.0
                    ),
                }
            )
    audit, _ = build_event_eligibility(curve_history, pd.DataFrame(rows))
    first = audit.loc[audit["event_date"].eq(exact_date)].iloc[0]
    second = audit.loc[audit["event_date"].eq(missing_date)].iloc[0]
    assert bool(first["eligible"])
    assert not bool(second["eligible"])
    assert second["missing_shock_maturities"] == "10Y"
    assert not bool(audit["future_curve_observations_used"].any())
