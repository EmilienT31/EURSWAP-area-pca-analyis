"""Reproducible sovereign-proxy swap valuation and EA-MPD event repricing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ecb_curve_data import curve_on_date, load_cached_aaa_curve
from .swap_pricing import (
    OISSwap,
    bootstrap_par_swap_curve,
    bp_to_decimal,
    key_rate_sensitivities,
    parallel_dv01,
)


NOTIONAL_EUR = 10_000_000.0
SWAP_MATURITIES = (2.0, 5.0, 10.0)
BOOTSTRAP_MATURITIES = tuple(float(value) for value in range(1, 11))
KEY_RATE_MATURITIES = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
EVENT_WINDOW = "Monetary Event Window"


def _load_event_shocks(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["event_date"])
    required = {
        "event_date",
        "window",
        "canonical_name",
        "maturity_years",
        "surprise_bp",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"EA-MPD surprise file is missing columns: {missing}.")
    annual = frame.loc[
        frame["window"].eq(EVENT_WINDOW)
        & frame["maturity_years"].isin(BOOTSTRAP_MATURITIES),
        ["event_date", "maturity_years", "surprise_bp"],
    ].copy()
    return annual


def _select_latest_complete_event(
    surprises: pd.DataFrame, available_curve_dates: pd.Series
) -> tuple[pd.Timestamp, pd.Series]:
    pivot = surprises.pivot(
        index="event_date", columns="maturity_years", values="surprise_bp"
    ).reindex(columns=BOOTSTRAP_MATURITIES)
    complete = pivot.dropna()
    curve_dates = set(pd.to_datetime(available_curve_dates).dt.normalize())
    eligible = complete.loc[complete.index.normalize().isin(curve_dates)]
    if eligible.empty:
        raise ValueError(
            "No event has both a complete annual 1Y-10Y EA-MPD shock and an ECB curve."
        )
    date = eligible.index.max()
    return pd.Timestamp(date), eligible.loc[date]


def _plot_curves(
    event_date: pd.Timestamp,
    base_curve,
    shocked_curve,
    base_par_rates: np.ndarray,
    shocked_par_rates: np.ndarray,
    destination: Path,
) -> None:
    maturities = np.asarray(BOOTSTRAP_MATURITIES)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    palette = {"initial": "#125B8C", "shocked": "#C55A11"}
    axes[0].plot(
        maturities,
        np.asarray(base_curve.zero_rate(maturities)) * 10_000,
        marker="o",
        color=palette["initial"],
        label="Initial AAA sovereign proxy",
    )
    axes[0].plot(
        maturities,
        np.asarray(shocked_curve.zero_rate(maturities)) * 10_000,
        marker="s",
        color=palette["shocked"],
        label="Rebootstrapped after EA-MPD shock",
    )
    axes[0].set_title("Continuously compounded zero rates")
    axes[0].set_ylabel("Rate (basis points)")
    axes[0].set_xlabel("Maturity (years)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        maturities,
        base_par_rates * 10_000,
        marker="o",
        color=palette["initial"],
        label="Initial implied par curve",
    )
    axes[1].plot(
        maturities,
        shocked_par_rates * 10_000,
        marker="s",
        color=palette["shocked"],
        label="Initial par curve + OIS surprise",
    )
    axes[1].set_title("Stylised annual-pay par swap rates")
    axes[1].set_ylabel("Rate (basis points)")
    axes[1].set_xlabel("Maturity (years)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle(f"Initial and counterfactual shocked curves — {event_date.date()}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _validation_table(
    swaps: list[OISSwap],
    base_curve,
    base_par_rates: np.ndarray,
    bootstrapped_base,
) -> pd.DataFrame:
    par_pvs = np.asarray([swap.value(base_curve) for swap in swaps])
    opposite_errors = []
    upward_changes = []
    key_parallel_errors = []
    for payer in swaps:
        receiver = OISSwap(
            payer.notional,
            payer.fixed_rate,
            payer.maturity_years,
            "receiver",
            payer.payment_frequency,
        )
        opposite_errors.append(abs(payer.value(base_curve) + receiver.value(base_curve)))
        upward_changes.append(payer.value(base_curve.parallel_bump(1.0)) - payer.value(base_curve))
        keys = key_rate_sensitivities(
            payer, base_curve, KEY_RATE_MATURITIES, bump_bp=1.0
        )
        key_parallel_errors.append(
            abs(keys["sensitivity_eur_per_bp"].sum() - parallel_dv01(payer, base_curve))
        )
    reproduced = np.asarray(
        [
            bootstrapped_base.par_swap_rate(maturity)
            for maturity in BOOTSTRAP_MATURITIES
        ]
    )
    rows = [
        {
            "check": "par_swap_pv_is_zero",
            "metric": float(np.max(np.abs(par_pvs))),
            "units": "EUR absolute error",
            "tolerance": 1.0e-6,
            "passed": bool(np.max(np.abs(par_pvs)) <= 1.0e-6),
        },
        {
            "check": "payer_receiver_are_opposites",
            "metric": float(max(opposite_errors)),
            "units": "EUR absolute error",
            "tolerance": 1.0e-9,
            "passed": bool(max(opposite_errors) <= 1.0e-9),
        },
        {
            "check": "payer_gains_when_curve_rises",
            "metric": float(min(upward_changes)),
            "units": "minimum EUR gain for +1 bp",
            "tolerance": 0.0,
            "passed": bool(min(upward_changes) > 0.0),
        },
        {
            "check": "one_bp_decimal_conversion",
            "metric": bp_to_decimal(1.0),
            "units": "decimal rate",
            "tolerance": 0.0001,
            "passed": bool(bp_to_decimal(1.0) == 0.0001),
        },
        {
            "check": "bootstrap_reproduces_par_quotes",
            "metric": float(np.max(np.abs(reproduced - base_par_rates)) * 10_000),
            "units": "basis points maximum error",
            "tolerance": 1.0e-8,
            "passed": bool(
                np.max(np.abs(reproduced - base_par_rates)) * 10_000 <= 1.0e-8
            ),
        },
        {
            "check": "key_rates_sum_to_parallel_first_order",
            "metric": float(max(key_parallel_errors)),
            "units": "EUR absolute difference per bp",
            "tolerance": 0.25,
            "passed": bool(max(key_parallel_errors) <= 0.25),
        },
    ]
    return pd.DataFrame(rows)


def _methodology(
    event_date: pd.Timestamp,
    curve_path: Path,
    event_shocks: pd.Series,
    validation: pd.DataFrame,
) -> str:
    passed = int(validation["passed"].sum())
    total = len(validation)
    return f"""# Minimal EUR OIS pricing and curve-risk methodology

## Data and event

The initial curve is loaded from `{curve_path}` and contains only ECB series whose instrument code is `G_N_A`: AAA-rated euro-area central-government bonds. The selected snapshot is **{event_date.date()}**, the latest date with both a complete annual 1Y–10Y EA-MPD Monetary Event Window shock and all requested ECB curve nodes. ECB government curves are normally published at noon; treating the same-day curve as pre-announcement is an operational proxy assumption, not a historical dealer mark.

The local cache contains 3M, 6M, 9M and annual 1Y–10Y continuously compounded spot rates. It is deliberately small; the original API payload is hashed but not retained. The earlier all-ratings bulk file is not used.

The MMSR dashboard is not used for bootstrapping. Its OIS rates are transaction-volume-weighted averages over reserve-maintenance periods and broad maturity buckets, rather than simultaneous exact-tenor par quotations.

## Curve conventions

- Model time and day count: ACT/365F year fractions.
- Zero-rate compounding: continuously compounded, so `D(t) = exp(-z(t)t)`.
- Interpolation: linear in log discount factors, equivalent to piecewise-constant instantaneous forwards.
- Extrapolation: prohibited beyond the final 10Y node.
- The initial ECB curve is a **sovereign-curve proxy**, not an €STR OIS discount curve.

## Swap conventions

Swaps are spot-starting fixed-for-overnight contracts with €10 million notional, annual payments, unit annual accruals, one projection/discount curve, no payment lags, no stubs and no business-day adjustment. The floating-leg identity is `N(1-D(T))`; the fixed leg is `N K sum(D(t_i))`. A payer pays fixed and receives floating, so a par payer gains when rates rise.

`parallel_dv01_eur_per_bp` is the central first-order PV change for a **+1 bp shift in every continuously compounded zero node**. Key-rate sensitivities use triangular zero-rate basis functions whose weights sum to one across the curve.

## EA-MPD counterfactual shock

For {event_date.date()}, the observed annual OIS changes range from **{event_shocks.min():.2f} bp to {event_shocks.max():.2f} bp**. They are converted with `1 bp = 0.0001`, added to the proxy curve's implied **par swap rates**, and the shocked discount curve is re-bootstrapped. OIS changes are never added directly to zero rates. No event observations are imputed; the selected event is complete at every 1Y–10Y node.

Existing 2Y, 5Y and 10Y swaps retain their initial par coupons when repriced on the shocked curve. Full-repricing P&L therefore measures the value change of those fixed coupons under this stylised counterfactual curve.

## Output interpretation

- `swap_valuation_and_dv01.csv`: initial par coupons, leg PVs, zero initial net PV and directional parallel risk. Longer swaps normally have larger absolute first-order exposure.
- `key_rate_sensitivities.csv`: allocation of each swap's first-order curve exposure across the common 1Y, 2Y, 3Y, 5Y, 7Y and 10Y zero-rate bump basis.
- `event_pnl_comparison.csv`: full nonlinear event repricing versus a simple parallel-DV01 approximation using the EA-MPD shock at the swap's own maturity. Differences reflect the non-parallel shock shape and convexity.
- `initial_and_shocked_curves.png`: initial and counterfactual zero/par curves in basis points. The initial curve is observed sovereign data; the shocked curve is model-generated.
- `validation_table.csv`: {passed} of {total} pricing invariants pass using the documented tolerances.

## Limitations

The exercise is linear single-curve risk, not production OIS valuation. It omits actual cash-flow dates, TARGET calendars, ACT/360 market accrual, payment delays, collateral agreements, bid/offer spreads, multi-curve effects and dealer curve construction. Historical overnight benchmarks transitioned from EONIA to €STR; the EA-MPD sample spans that transition. The AAA government curve carries sovereign and bond-market effects and must not be described as an OIS curve. Applying OIS event changes to its implied par rates produces a transparent counterfactual P&L, not a reconstructed historical dealer mark or an independent causal estimate.
"""


def run(curve_path: Path, surprises_path: Path, output_dir: Path) -> None:
    curve_history = load_cached_aaa_curve(curve_path)
    surprises = _load_event_shocks(surprises_path)
    event_date, event_shocks = _select_latest_complete_event(
        surprises, curve_history["date"]
    )
    base_curve = curve_on_date(curve_history, event_date)
    base_par_rates = np.asarray(
        [base_curve.par_swap_rate(maturity) for maturity in BOOTSTRAP_MATURITIES]
    )
    event_shock_bp = event_shocks.reindex(BOOTSTRAP_MATURITIES).to_numpy(float)
    shocked_par_rates = base_par_rates + event_shock_bp * 1.0e-4
    shocked_curve = bootstrap_par_swap_curve(
        BOOTSTRAP_MATURITIES, shocked_par_rates
    )
    bootstrapped_base = bootstrap_par_swap_curve(
        BOOTSTRAP_MATURITIES, base_par_rates
    )

    swaps = [
        OISSwap.at_par(base_curve, NOTIONAL_EUR, maturity, side="payer")
        for maturity in SWAP_MATURITIES
    ]
    valuation_rows: list[dict[str, float]] = []
    pnl_rows: list[dict[str, float | str]] = []
    key_frames: list[pd.DataFrame] = []
    for swap in swaps:
        receiver = OISSwap(
            swap.notional,
            swap.fixed_rate,
            swap.maturity_years,
            side="receiver",
        )
        dv01 = parallel_dv01(swap, base_curve)
        valuation_rows.append(
            {
                "maturity_years": swap.maturity_years,
                "notional_eur": swap.notional,
                "par_fixed_rate_decimal": swap.fixed_rate,
                "par_fixed_rate_bp": swap.fixed_rate * 10_000,
                "fixed_leg_pv_eur": swap.fixed_leg_pv(base_curve),
                "floating_leg_pv_eur": swap.floating_leg_pv(base_curve),
                "payer_pv_eur": swap.value(base_curve),
                "receiver_pv_eur": receiver.value(base_curve),
                "payer_parallel_dv01_eur_per_bp": dv01,
                "receiver_parallel_dv01_eur_per_bp": -dv01,
            }
        )
        shock_bp = float(event_shocks.loc[swap.maturity_years])
        full_pnl = swap.value(shocked_curve) - swap.value(base_curve)
        parallel_approximation = dv01 * shock_bp
        pnl_rows.append(
            {
                "event_date": event_date.date().isoformat(),
                "maturity_years": swap.maturity_years,
                "ois_surprise_at_swap_maturity_bp": shock_bp,
                "full_repricing_pnl_eur": full_pnl,
                "parallel_dv01_approximation_eur": parallel_approximation,
                "full_minus_approximation_eur": full_pnl - parallel_approximation,
            }
        )
        keys = key_rate_sensitivities(swap, base_curve, KEY_RATE_MATURITIES)
        keys.insert(0, "swap_maturity_years", swap.maturity_years)
        keys["bump_definition"] = "triangular +1 bp continuously compounded zero rate"
        key_frames.append(keys)

    valuation = pd.DataFrame(valuation_rows)
    pnl = pd.DataFrame(pnl_rows)
    key_rates = pd.concat(key_frames, ignore_index=True)
    validation = _validation_table(
        swaps, base_curve, base_par_rates, bootstrapped_base
    )
    if not validation["passed"].all():
        failed = validation.loc[~validation["passed"], "check"].tolist()
        raise RuntimeError(f"Pricing validation failed: {failed}")

    output_dir.mkdir(parents=True, exist_ok=True)
    valuation.to_csv(output_dir / "swap_valuation_and_dv01.csv", index=False)
    pnl.to_csv(output_dir / "event_pnl_comparison.csv", index=False)
    key_rates.to_csv(output_dir / "key_rate_sensitivities.csv", index=False)
    validation.to_csv(output_dir / "validation_table.csv", index=False)
    curve_nodes = pd.DataFrame(
        {
            "event_date": event_date.date().isoformat(),
            "maturity_years": BOOTSTRAP_MATURITIES,
            "initial_zero_rate_bp": np.asarray(
                base_curve.zero_rate(BOOTSTRAP_MATURITIES)
            )
            * 10_000,
            "initial_par_rate_bp": base_par_rates * 10_000,
            "ois_surprise_bp": event_shock_bp,
            "shocked_par_rate_bp": shocked_par_rates * 10_000,
            "rebootstrapped_shocked_zero_rate_bp": shocked_curve.zero_rates_cc
            * 10_000,
        }
    )
    curve_nodes.to_csv(output_dir / "event_curve_nodes.csv", index=False)
    _plot_curves(
        event_date,
        base_curve,
        shocked_curve,
        base_par_rates,
        shocked_par_rates,
        output_dir / "initial_and_shocked_curves.png",
    )
    (output_dir / "methodology.md").write_text(
        _methodology(event_date, curve_path, event_shocks, validation),
        encoding="utf-8",
    )
    metadata = {
        "event_date": event_date.date().isoformat(),
        "event_window": EVENT_WINDOW,
        "curve_source": str(curve_path),
        "surprise_source": str(surprises_path),
        "notional_eur": NOTIONAL_EUR,
        "swap_maturities_years": SWAP_MATURITIES,
        "bootstrap_maturities_years": BOOTSTRAP_MATURITIES,
        "key_rate_maturities_years": KEY_RATE_MATURITIES,
        "rate_units_in_curve_table": "basis points",
        "pnl_units": "EUR",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Price stylised EUR OIS swaps and apply one real EA-MPD event shock."
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
