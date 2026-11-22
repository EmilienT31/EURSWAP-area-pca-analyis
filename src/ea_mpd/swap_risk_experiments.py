"""Event-level comparison of par, parallel-zero and key-rate swap risk."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .ecb_curve_data import SPOT_TYPES, curve_on_date, load_cached_aaa_curve
from .swap_pricing import (
    OISSwap,
    YieldCurve,
    bootstrap_par_swap_curve,
    key_rate_sensitivities,
    parallel_dv01,
    par_rate_annuity_sensitivity,
)


EVENT_WINDOW = "Monetary Event Window"
NODE_GRID = tuple(float(value) for value in range(1, 11))
SWAP_MATURITIES = (2.0, 5.0, 10.0)
BASE_NOTIONAL_EUR = 10_000_000.0
REFERENCE_EVENT = pd.Timestamp("2025-10-30")


@dataclass(frozen=True)
class EventCurveState:
    """Initial and shocked curves on the common annual zero-node basis."""

    event_date: pd.Timestamp
    base_curve: YieldCurve
    shocked_curve: YieldCurve
    base_par_rates: np.ndarray
    shocked_par_rates: np.ndarray
    par_shocks_bp: np.ndarray
    zero_shocks_bp: np.ndarray
    parallel_zero_shock_bp: float


def load_annual_event_shocks(path: str | Path) -> pd.DataFrame:
    """Load Monetary Event Window shocks without dropping missing observations."""

    frame = pd.read_csv(path, parse_dates=["event_date"])
    required = {"event_date", "window", "maturity_years", "surprise_bp"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"EA-MPD surprise file is missing columns: {missing}.")
    result = frame.loc[
        frame["window"].eq(EVENT_WINDOW)
        & frame["maturity_years"].isin(NODE_GRID),
        ["event_date", "maturity_years", "surprise_bp"],
    ].copy()
    duplicates = result.duplicated(["event_date", "maturity_years"], keep=False)
    if duplicates.any():
        raise ValueError("EA-MPD surprises contain duplicate event/maturity rows.")
    return result.sort_values(["event_date", "maturity_years"], ignore_index=True)


def build_event_eligibility(
    curve_history: pd.DataFrame, annual_shocks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit exact-date curve alignment and complete 1Y--10Y shock coverage."""

    pivot = annual_shocks.pivot(
        index="event_date", columns="maturity_years", values="surprise_bp"
    ).reindex(columns=NODE_GRID)
    curve_counts = curve_history.groupby(curve_history["date"].dt.normalize()).agg(
        curve_nodes=("maturity_years", "nunique"),
        available_rates=("zero_rate_decimal", "count"),
    )
    complete_curve_dates = set(
        curve_counts.index[
            (curve_counts["curve_nodes"] == len(SPOT_TYPES))
            & (curve_counts["available_rates"] == len(SPOT_TYPES))
        ]
    )
    rows: list[dict[str, object]] = []
    for event_date, shocks in pivot.iterrows():
        normalized = pd.Timestamp(event_date).normalize()
        observed = int(shocks.notna().sum())
        missing_terms = [f"{int(term)}Y" for term in NODE_GRID if pd.isna(shocks[term])]
        has_curve = normalized in complete_curve_dates
        complete_shocks = observed == len(NODE_GRID)
        eligible = has_curve and complete_shocks
        reasons: list[str] = []
        if not has_curve:
            reasons.append("no complete exact-date public curve")
        if not complete_shocks:
            reasons.append("missing required annual EA-MPD shocks")
        rows.append(
            {
                "event_date": normalized,
                "curve_date_used": normalized if has_curve else pd.NaT,
                "date_alignment_days": 0 if has_curve else pd.NA,
                "curve_alignment_rule": "exact same TARGET date only",
                "curve_time_assumption": (
                    "ECB noon proxy, before policy announcement" if has_curve else ""
                ),
                "required_shock_grid": "1Y,2Y,3Y,4Y,5Y,6Y,7Y,8Y,9Y,10Y",
                "required_shocks_observed": observed,
                "missing_shock_maturities": ";".join(missing_terms),
                "complete_required_shocks": complete_shocks,
                "complete_exact_date_curve": has_curve,
                "eligible": eligible,
                "exclusion_reason": "; ".join(reasons),
                "future_curve_observations_used": False,
                "shock_imputation_used": False,
            }
        )
    audit = pd.DataFrame(rows).sort_values("event_date", ignore_index=True)
    return audit, pivot


def build_event_curve_state(
    curve_history: pd.DataFrame,
    event_date: pd.Timestamp,
    par_shocks: pd.Series,
) -> EventCurveState:
    """Rebootstrap initial and shocked annual-node curves for one event."""

    shocks = par_shocks.reindex(NODE_GRID)
    if shocks.isna().any():
        missing = [f"{int(term)}Y" for term in NODE_GRID if pd.isna(shocks[term])]
        raise ValueError(f"Event {event_date.date()} has missing shocks: {missing}.")
    observed_proxy = curve_on_date(curve_history, event_date)
    base_par_rates = np.asarray(
        [observed_proxy.par_swap_rate(term) for term in NODE_GRID]
    )
    base_curve = bootstrap_par_swap_curve(NODE_GRID, base_par_rates)
    shock_bp = shocks.to_numpy(float)
    shocked_par_rates = base_par_rates + shock_bp * 1.0e-4
    shocked_curve = bootstrap_par_swap_curve(NODE_GRID, shocked_par_rates)
    zero_shocks_bp = (
        shocked_curve.zero_rates_cc - base_curve.zero_rates_cc
    ) * 10_000
    return EventCurveState(
        pd.Timestamp(event_date).normalize(),
        base_curve,
        shocked_curve,
        base_par_rates,
        shocked_par_rates,
        shock_bp,
        zero_shocks_bp,
        float(np.mean(zero_shocks_bp)),
    )


def zero_risk_decomposition(
    key_sensitivities_eur_per_bp: Iterable[float],
    parallel_sensitivity_eur_per_bp: float,
    zero_shocks_bp: Iterable[float],
) -> dict[str, float]:
    """Split annual-node KRD P&L into mean-parallel and residual-shape pieces."""

    krd = np.asarray(tuple(key_sensitivities_eur_per_bp), dtype=float)
    shocks = np.asarray(tuple(zero_shocks_bp), dtype=float)
    if krd.shape != shocks.shape or krd.ndim != 1 or len(krd) == 0:
        raise ValueError("KRDs and zero shocks must be equal non-empty vectors.")
    parallel_shift = float(np.mean(shocks))
    parallel_pnl = parallel_sensitivity_eur_per_bp * parallel_shift
    shape_pnl = float(np.dot(krd, shocks - parallel_shift))
    krd_pnl = float(np.dot(krd, shocks))
    return {
        "parallel_zero_shock_bp": parallel_shift,
        "parallel_only_approximation_eur": parallel_pnl,
        "residual_shape_krd_approximation_eur": shape_pnl,
        "zero_krd_approximation_eur": krd_pnl,
        "krd_minus_parallel_plus_shape_eur": krd_pnl - parallel_pnl - shape_pnl,
    }


def _swap_profile(swap: OISSwap, state: EventCurveState) -> dict[str, object]:
    krd_table = key_rate_sensitivities(swap, state.base_curve, NODE_GRID)
    krd = krd_table["sensitivity_eur_per_bp"].to_numpy(float)
    parallel = parallel_dv01(swap, state.base_curve)
    annuity_per_bp = par_rate_annuity_sensitivity(swap, state.base_curve, per_bp=True)
    maturity_index = NODE_GRID.index(swap.maturity_years)
    par_shock = float(state.par_shocks_bp[maturity_index])
    decomposition = zero_risk_decomposition(
        krd, parallel, state.zero_shocks_bp
    )
    full_pnl = swap.value(state.shocked_curve) - swap.value(state.base_curve)
    par_approximation = annuity_per_bp * par_shock
    return {
        "event_date": state.event_date,
        "curve_date_used": state.event_date,
        "date_alignment_days": 0,
        "instrument": f"{int(swap.maturity_years)}Y_{swap.side}",
        "side": swap.side,
        "maturity_years": swap.maturity_years,
        "notional_eur": swap.notional,
        "fixed_coupon_bp": swap.fixed_rate * 10_000,
        "own_par_shock_bp": par_shock,
        "par_rate_annuity_sensitivity_eur_per_decimal": par_rate_annuity_sensitivity(
            swap, state.base_curve, per_bp=False
        ),
        "par_rate_annuity_sensitivity_eur_per_bp": annuity_per_bp,
        "parallel_zero_dv01_eur_per_bp": parallel,
        "sum_zero_node_krd_eur_per_bp": float(np.sum(krd)),
        "own_maturity_zero_krd_eur_per_bp": float(krd[maturity_index]),
        "full_repricing_pnl_eur": full_pnl,
        "matched_par_annuity_approximation_eur": par_approximation,
        **decomposition,
        "full_minus_par_annuity_eur": full_pnl - par_approximation,
        "full_minus_parallel_only_eur": full_pnl
        - decomposition["parallel_only_approximation_eur"],
        "full_minus_zero_krd_eur": full_pnl
        - decomposition["zero_krd_approximation_eur"],
        "portfolio_10y_receiver_notional_eur": np.nan,
        "future_curve_observations_used": False,
        "shock_imputation_used": False,
        "_krd": krd,
    }


def dv01_neutral_curve_portfolio(state: EventCurveState) -> tuple[OISSwap, OISSwap]:
    """Build 10mm 2Y payer plus 10Y receiver sized to cancel parallel zero DV01."""

    payer_2y = OISSwap.at_par(
        state.base_curve, BASE_NOTIONAL_EUR, 2.0, side="payer"
    )
    receiver_10y_reference = OISSwap.at_par(
        state.base_curve, BASE_NOTIONAL_EUR, 10.0, side="receiver"
    )
    payer_dv01 = parallel_dv01(payer_2y, state.base_curve)
    receiver_dv01 = parallel_dv01(receiver_10y_reference, state.base_curve)
    receiver_notional = BASE_NOTIONAL_EUR * payer_dv01 / abs(receiver_dv01)
    receiver_10y = OISSwap.at_par(
        state.base_curve, receiver_notional, 10.0, side="receiver"
    )
    return payer_2y, receiver_10y


def _portfolio_profile(
    state: EventCurveState,
    payer_2y: OISSwap,
    receiver_10y: OISSwap,
    leg_profiles: dict[str, dict[str, object]],
) -> dict[str, object]:
    payer_profile = leg_profiles["2Y_payer"]
    receiver_krd = key_rate_sensitivities(
        receiver_10y, state.base_curve, NODE_GRID
    )["sensitivity_eur_per_bp"].to_numpy(float)
    receiver_parallel = parallel_dv01(receiver_10y, state.base_curve)
    receiver_annuity = par_rate_annuity_sensitivity(
        receiver_10y, state.base_curve, per_bp=True
    )
    receiver_full = receiver_10y.value(state.shocked_curve) - receiver_10y.value(
        state.base_curve
    )
    receiver_par = receiver_annuity * state.par_shocks_bp[NODE_GRID.index(10.0)]

    portfolio_krd = np.asarray(payer_profile["_krd"]) + receiver_krd
    portfolio_parallel = (
        float(payer_profile["parallel_zero_dv01_eur_per_bp"])
        + receiver_parallel
    )
    decomposition = zero_risk_decomposition(
        portfolio_krd, portfolio_parallel, state.zero_shocks_bp
    )
    full_pnl = float(payer_profile["full_repricing_pnl_eur"]) + receiver_full
    par_approximation = (
        float(payer_profile["matched_par_annuity_approximation_eur"])
        + receiver_par
    )
    return {
        "event_date": state.event_date,
        "curve_date_used": state.event_date,
        "date_alignment_days": 0,
        "instrument": "2Y_payer_10Y_receiver_dv01_neutral",
        "side": "curve_portfolio",
        "maturity_years": np.nan,
        "notional_eur": payer_2y.notional + receiver_10y.notional,
        "fixed_coupon_bp": np.nan,
        "own_par_shock_bp": np.nan,
        "par_rate_annuity_sensitivity_eur_per_decimal": np.nan,
        "par_rate_annuity_sensitivity_eur_per_bp": np.nan,
        "parallel_zero_dv01_eur_per_bp": portfolio_parallel,
        "sum_zero_node_krd_eur_per_bp": float(np.sum(portfolio_krd)),
        "own_maturity_zero_krd_eur_per_bp": np.nan,
        "full_repricing_pnl_eur": full_pnl,
        "matched_par_annuity_approximation_eur": par_approximation,
        **decomposition,
        "full_minus_par_annuity_eur": full_pnl - par_approximation,
        "full_minus_parallel_only_eur": full_pnl
        - decomposition["parallel_only_approximation_eur"],
        "full_minus_zero_krd_eur": full_pnl
        - decomposition["zero_krd_approximation_eur"],
        "portfolio_10y_receiver_notional_eur": receiver_10y.notional,
        "future_curve_observations_used": False,
        "shock_imputation_used": False,
        "_krd": portfolio_krd,
    }


def _event_rows(state: EventCurveState) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    swaps = [
        OISSwap.at_par(state.base_curve, BASE_NOTIONAL_EUR, term, side="payer")
        for term in SWAP_MATURITIES
    ]
    profiles = [_swap_profile(swap, state) for swap in swaps]
    by_instrument = {str(row["instrument"]): row for row in profiles}
    payer_2y, receiver_10y = dv01_neutral_curve_portfolio(state)
    portfolio = _portfolio_profile(
        state, payer_2y, receiver_10y, by_instrument
    )
    profiles.append(portfolio)

    zero_rows = [
        {
            "event_date": state.event_date,
            "curve_date_used": state.event_date,
            "date_alignment_days": 0,
            "maturity_years": maturity,
            "initial_zero_rate_bp": state.base_curve.zero_rates_cc[index] * 10_000,
            "rebootstrapped_shocked_zero_rate_bp": state.shocked_curve.zero_rates_cc[index]
            * 10_000,
            "zero_rate_change_bp": state.zero_shocks_bp[index],
            "parallel_component_bp": state.parallel_zero_shock_bp,
            "shape_component_bp": state.zero_shocks_bp[index]
            - state.parallel_zero_shock_bp,
            "initial_par_rate_bp": state.base_par_rates[index] * 10_000,
            "ois_par_shock_bp": state.par_shocks_bp[index],
            "shocked_par_rate_bp": state.shocked_par_rates[index] * 10_000,
        }
        for index, maturity in enumerate(NODE_GRID)
    ]
    return profiles, zero_rows


def approximation_metrics(event_results: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSE and MAE against full repricing for each approximation."""

    methods = {
        "matched_par_annuity": "matched_par_annuity_approximation_eur",
        "parallel_zero_only": "parallel_only_approximation_eur",
        "annual_zero_node_krd": "zero_krd_approximation_eur",
    }
    rows: list[dict[str, object]] = []
    instruments = [*sorted(event_results["instrument"].unique()), "ALL"]
    for instrument in instruments:
        subset = (
            event_results
            if instrument == "ALL"
            else event_results.loc[event_results["instrument"].eq(instrument)]
        )
        actual = subset["full_repricing_pnl_eur"].to_numpy(float)
        for method, column in methods.items():
            error = subset[column].to_numpy(float) - actual
            rows.append(
                {
                    "instrument": instrument,
                    "approximation": method,
                    "n_events": len(subset),
                    "rmse_eur": float(np.sqrt(np.mean(error**2))),
                    "mae_eur": float(np.mean(np.abs(error))),
                    "mean_error_eur": float(np.mean(error)),
                    "max_absolute_error_eur": float(np.max(np.abs(error))),
                }
            )
    return pd.DataFrame(rows)


def _portfolio_scenarios(
    state: EventCurveState,
    payer_2y: OISSwap,
    receiver_10y: OISSwap,
) -> pd.DataFrame:
    portfolio_krd = (
        key_rate_sensitivities(payer_2y, state.base_curve, NODE_GRID)[
            "sensitivity_eur_per_bp"
        ].to_numpy(float)
        + key_rate_sensitivities(receiver_10y, state.base_curve, NODE_GRID)[
            "sensitivity_eur_per_bp"
        ].to_numpy(float)
    )
    parallel = parallel_dv01(payer_2y, state.base_curve) + parallel_dv01(
        receiver_10y, state.base_curve
    )

    def portfolio_value(curve: YieldCurve) -> float:
        return payer_2y.value(curve) + receiver_10y.value(curve)

    scenarios: list[tuple[str, np.ndarray, YieldCurve]] = []
    parallel_shock = np.ones(len(NODE_GRID))
    scenarios.append(
        ("uniform_zero_up_1bp", parallel_shock, state.base_curve.parallel_bump(1.0))
    )
    for label, key in [("2y_zero_key_up_1bp", 2.0), ("10y_zero_key_up_1bp", 10.0)]:
        shock = np.zeros(len(NODE_GRID))
        shock[NODE_GRID.index(key)] = 1.0
        scenarios.append(
            (label, shock, state.base_curve.key_rate_bump(key, 1.0, NODE_GRID))
        )
    scenarios.append(
        ("observed_2025_10_30_event", state.zero_shocks_bp, state.shocked_curve)
    )
    rows = []
    base_value = portfolio_value(state.base_curve)
    for label, shock, shocked_curve in scenarios:
        decomposition = zero_risk_decomposition(portfolio_krd, parallel, shock)
        rows.append(
            {
                "scenario": label,
                "arithmetic_mean_zero_shock_bp": float(np.mean(shock)),
                "full_repricing_pnl_eur": portfolio_value(shocked_curve) - base_value,
                **decomposition,
            }
        )
    return pd.DataFrame(rows)


def _methodology(
    audit: pd.DataFrame,
    event_results: pd.DataFrame,
    metrics: pd.DataFrame,
) -> str:
    eligible = audit.loc[audit["eligible"]]
    start = eligible["event_date"].min().date()
    end = eligible["event_date"].max().date()
    reference = event_results.loc[event_results["event_date"].eq(REFERENCE_EVENT)]
    portfolio = reference.loc[
        reference["instrument"].eq("2Y_payer_10Y_receiver_dv01_neutral")
    ].iloc[0]
    metric_lines = []
    for method in ("matched_par_annuity", "parallel_zero_only", "annual_zero_node_krd"):
        row = metrics.loc[
            metrics["instrument"].eq("ALL") & metrics["approximation"].eq(method)
        ].iloc[0]
        metric_lines.append(
            f"- `{method}`: RMSE €{row['rmse_eur']:,.2f}; MAE €{row['mae_eur']:,.2f}."
        )
    metrics_text = "\n".join(metric_lines)
    return f"""# Swap-risk decomposition and event-level approximation study

## Samples and alignment

The audit begins with all **{len(audit)}** EA-MPD Monetary Event Window dates. An event is eligible only when (i) the public ECB AAA sovereign proxy has all requested nodes on the exact event date and (ii) EA-MPD reports every annual 1Y–10Y OIS par-rate change. No previous-day or next-day curve is substituted. No shock is imputed. This leaves **{len(eligible)} events**, from **{start} to {end}**. `event_alignment_and_eligibility.csv` retains excluded dates, missing maturities and the reason for exclusion.

For eligible dates, the ECB curve is the same-date noon sovereign proxy, which precedes the usual afternoon policy announcement. Each event's swap coupons, risk measures and curve-portfolio hedge ratio are computed from that date's initial curve only. No later curve observation or future hedge ratio is used.

## Three distinct risk measures

1. **Matched par-rate annuity sensitivity:** for a par swap, `direction × N × A0` per unit decimal change in its own par quote, or `direction × N × A0 × 0.0001` per bp. It is a quote-space sensitivity and assumes the initial annuity for first-order P&L.
2. **Parallel zero-rate DV01:** central PV change when every continuously compounded zero node is shifted by +1 bp. It is a uniform zero-space scenario.
3. **Zero-node key-rate sensitivities:** central PV changes for +1 bp at individual annual zero nodes, with log-discount interpolation between nodes. Their sum is consistent with the parallel zero DV01, but no individual KRD is interchangeable with either the par-annuity sensitivity or the parallel measure.

`risk_measure_comparison_2025-10-30.csv` reports all three for 2Y, 5Y and 10Y payer swaps. Their numerical differences arise because a par quote is a weighted function of multiple discount factors, while zero-node bumps directly alter particular discount factors.

## Event P&L approximations

The observed EA-MPD annual par-rate changes are added to the initial implied par quotes and a shocked annual discount curve is re-bootstrapped. For each existing swap, whose fixed coupon remains unchanged, the study reports:

- full repricing;
- matched par approximation `direction × N × A0 × delta_S`;
- annual-zero-node KRD approximation `sum_i KRD_i × delta_z_i`;
- parallel-only approximation `parallel zero DV01 × mean(delta_z)`.

The fixed parallel component is the arithmetic mean of the 1Y, 2Y, ..., 10Y continuously compounded zero-rate changes. Shape changes are `delta_z_i - mean(delta_z)`. The residual shape approximation is `sum_i KRD_i × shape_i`; parallel plus shape reconstructs the KRD approximation up to the small finite-bump consistency error.

Across all instruments and eligible events:

{metrics_text}

These errors compare approximations with model full repricing; they do not validate the sovereign proxy against dealer OIS marks.

## DV01-neutral curve portfolio

At each event, the portfolio pays fixed in a €10 million 2Y swap and receives fixed in a 10Y swap. The 10Y notional is determined from that event's initial parallel zero DV01 so the two signed parallel sensitivities approximately cancel. On 2025-10-30 the 10Y receiver notional is **€{portfolio['portfolio_10y_receiver_notional_eur']:,.2f}** and the remaining parallel DV01 is **€{portfolio['parallel_zero_dv01_eur_per_bp']:.6f} per bp**.

Parallel neutrality does not imply zero curve risk. `curve_portfolio_scenarios_2025-10-30.csv` shows near-zero P&L for a uniform 1 bp shift but material, opposite-signed exposure to isolated 2Y and 10Y node shifts and to the observed non-parallel event. The portfolio's parallel-only approximation is therefore intentionally uninformative about its shape exposure.

## Outputs and limitations

- `event_level_pnl_comparison.csv`: dated 2Y/5Y/10Y swap and dynamically hedged curve-portfolio results.
- `event_zero_curve_shocks.csv`: initial/shocked annual nodes and the mean/shape decomposition.
- `approximation_error_metrics.csv`: RMSE/MAE by instrument and approximation.
- `curve_portfolio_definition_2025-10-30.csv` and `curve_portfolio_scenarios_2025-10-30.csv`: hedge sizing and non-parallel scenario verification.

All original pricing limitations remain: annual ACT/365F model time, no actual schedules or payment lags, one discount/projection curve and a government-bond proxy rather than an €STR curve. The event exercise is a stylised counterfactual decomposition, not a causal test or historical dealer P&L. PCA P&L is deliberately outside this stage, and factor-regression results are untouched.
"""


def run(curve_path: Path, surprises_path: Path, output_dir: Path) -> None:
    curve_history = load_cached_aaa_curve(curve_path)
    annual_shocks = load_annual_event_shocks(surprises_path)
    audit, shock_pivot = build_event_eligibility(curve_history, annual_shocks)
    eligible_dates = audit.loc[audit["eligible"], "event_date"]

    event_rows: list[dict[str, object]] = []
    zero_rows: list[dict[str, object]] = []
    reference_state: EventCurveState | None = None
    for event_date in eligible_dates:
        state = build_event_curve_state(
            curve_history, pd.Timestamp(event_date), shock_pivot.loc[event_date]
        )
        rows, nodes = _event_rows(state)
        event_rows.extend(rows)
        zero_rows.extend(nodes)
        if state.event_date == REFERENCE_EVENT:
            reference_state = state
    if reference_state is None:
        raise RuntimeError(f"Reference event {REFERENCE_EVENT.date()} is not eligible.")

    event_results = pd.DataFrame(event_rows)
    hidden_columns = [column for column in event_results if column.startswith("_")]
    public_results = event_results.drop(columns=hidden_columns)
    zero_shocks = pd.DataFrame(zero_rows)
    metrics = approximation_metrics(public_results)

    reference = event_results.loc[event_results["event_date"].eq(REFERENCE_EVENT)]
    risk_comparison = reference.loc[
        reference["instrument"].isin(["2Y_payer", "5Y_payer", "10Y_payer"]),
        [
            "event_date",
            "instrument",
            "notional_eur",
            "par_rate_annuity_sensitivity_eur_per_bp",
            "parallel_zero_dv01_eur_per_bp",
            "sum_zero_node_krd_eur_per_bp",
            "own_maturity_zero_krd_eur_per_bp",
        ],
    ].copy()
    risk_comparison["annuity_minus_parallel_eur_per_bp"] = (
        risk_comparison["par_rate_annuity_sensitivity_eur_per_bp"]
        - risk_comparison["parallel_zero_dv01_eur_per_bp"]
    )
    risk_decomposition = public_results.loc[
        public_results["event_date"].eq(REFERENCE_EVENT)
    ].copy()

    payer_2y, receiver_10y = dv01_neutral_curve_portfolio(reference_state)
    portfolio_definition = pd.DataFrame(
        [
            {
                "event_date": REFERENCE_EVENT,
                "leg": "2Y payer",
                "maturity_years": 2.0,
                "side": "payer",
                "notional_eur": payer_2y.notional,
                "fixed_coupon_bp": payer_2y.fixed_rate * 10_000,
                "parallel_zero_dv01_eur_per_bp": parallel_dv01(
                    payer_2y, reference_state.base_curve
                ),
            },
            {
                "event_date": REFERENCE_EVENT,
                "leg": "10Y receiver",
                "maturity_years": 10.0,
                "side": "receiver",
                "notional_eur": receiver_10y.notional,
                "fixed_coupon_bp": receiver_10y.fixed_rate * 10_000,
                "parallel_zero_dv01_eur_per_bp": parallel_dv01(
                    receiver_10y, reference_state.base_curve
                ),
            },
        ]
    )
    portfolio_definition.loc[len(portfolio_definition)] = {
        "event_date": REFERENCE_EVENT,
        "leg": "portfolio total",
        "maturity_years": np.nan,
        "side": "curve portfolio",
        "notional_eur": portfolio_definition["notional_eur"].sum(),
        "fixed_coupon_bp": np.nan,
        "parallel_zero_dv01_eur_per_bp": portfolio_definition[
            "parallel_zero_dv01_eur_per_bp"
        ].sum(),
    }
    portfolio_scenarios = _portfolio_scenarios(
        reference_state, payer_2y, receiver_10y
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(
        output_dir / "event_alignment_and_eligibility.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    public_results.to_csv(
        output_dir / "event_level_pnl_comparison.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    zero_shocks.to_csv(
        output_dir / "event_zero_curve_shocks.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    metrics.to_csv(output_dir / "approximation_error_metrics.csv", index=False)
    risk_comparison.to_csv(
        output_dir / "risk_measure_comparison_2025-10-30.csv", index=False
    )
    risk_decomposition.to_csv(
        output_dir / "risk_decomposition_2025-10-30.csv", index=False
    )
    portfolio_definition.to_csv(
        output_dir / "curve_portfolio_definition_2025-10-30.csv", index=False
    )
    portfolio_scenarios.to_csv(
        output_dir / "curve_portfolio_scenarios_2025-10-30.csv", index=False
    )
    (output_dir / "risk_experiment_methodology.md").write_text(
        _methodology(audit, public_results, metrics), encoding="utf-8"
    )
    metadata = {
        "curve_source": str(curve_path),
        "surprise_source": str(surprises_path),
        "event_window": EVENT_WINDOW,
        "fixed_zero_node_grid_years": NODE_GRID,
        "parallel_component": "arithmetic mean of annual 1Y-10Y zero-rate changes",
        "total_event_dates_audited": len(audit),
        "eligible_event_dates": int(audit["eligible"].sum()),
        "first_eligible_event": eligible_dates.min().date().isoformat(),
        "last_eligible_event": eligible_dates.max().date().isoformat(),
        "future_observations_used": False,
        "missing_shocks_imputed": False,
    }
    (output_dir / "risk_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare par-annuity, parallel-zero and key-rate event P&L."
    )
    parser.add_argument(
        "--curve",
        type=Path,
        default=Path("data/external/ecb_aaa_curve/aaa_spot_rates_3m_10y.csv"),
    )
    parser.add_argument(
        "--surprises", type=Path, default=Path("results/ois_surprises.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/swap_pricing")
    )
    args = parser.parse_args()
    run(args.curve, args.surprises, args.output)


if __name__ == "__main__":
    main()
