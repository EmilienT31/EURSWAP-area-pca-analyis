"""Rolling historical-curve PCA attribution of ECB event swap P&L."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ecb_curve_data import load_cached_aaa_curve
from .swap_pricing import OISSwap, YieldCurve, key_rate_sensitivities
from .swap_risk_experiments import BASE_NOTIONAL_EUR, NODE_GRID


MIN_TRAINING_OBSERVATIONS = 250
MAX_TRAINING_OBSERVATIONS = 750
REFERENCE_EVENT = pd.Timestamp("2025-10-30")
METHOD_COLUMNS = {
    "parallel_zero_only": "parallel_only_approximation_eur",
    "pca_1": "pca_1_approximation_eur",
    "pca_2": "pca_2_approximation_eur",
    "pca_3": "pca_3_approximation_eur",
    "full_krd": "zero_krd_approximation_eur",
}


@dataclass(frozen=True)
class CovariancePCA:
    """Centered, unstandardized covariance PCA fitted to changes in bp."""

    mean_bp: np.ndarray
    loadings: np.ndarray
    eigenvalues_bp2: np.ndarray
    explained_variance_ratio: np.ndarray

    def scores(self, shock_bp: Iterable[float]) -> np.ndarray:
        shock = np.asarray(tuple(shock_bp), dtype=float)
        if shock.shape != self.mean_bp.shape:
            raise ValueError("Shock and PCA mean must have the same shape.")
        return (shock - self.mean_bp) @ self.loadings

    def reconstruct(self, shock_bp: Iterable[float], n_components: int) -> np.ndarray:
        if not 0 <= n_components <= self.loadings.shape[1]:
            raise ValueError("Invalid number of PCA components.")
        score = self.scores(shock_bp)
        return self.mean_bp + self.loadings[:, :n_components] @ score[:n_components]


def _orient_loading(loadings: np.ndarray) -> np.ndarray:
    """Apply deterministic economic sign conventions without changing PCA spans."""

    oriented = loadings.copy()
    maturities = np.asarray(NODE_GRID)
    for component in range(oriented.shape[1]):
        vector = oriented[:, component]
        if component == 0:
            statistic = float(np.mean(vector))
        elif component == 1:
            statistic = float(np.corrcoef(vector, maturities)[0, 1])
        elif component == 2:
            belly = float(np.mean(vector[3:7]))
            ends = float(np.mean(vector[[0, 1, 8, 9]]))
            statistic = belly - ends
        else:
            statistic = float(vector[np.argmax(np.abs(vector))])
        if statistic < 0:
            oriented[:, component] *= -1.0
    return oriented


def fit_covariance_pca(changes_bp: np.ndarray) -> CovariancePCA:
    """Fit PCA to a complete observations-by-maturities matrix in basis points."""

    values = np.asarray(changes_bp, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA requires at least two complete observations.")
    if values.shape[1] != len(NODE_GRID) or not np.isfinite(values).all():
        raise ValueError("PCA input must be a finite complete 1Y--10Y matrix.")
    mean = values.mean(axis=0)
    covariance = np.cov(values - mean, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    loadings = _orient_loading(eigenvectors[:, order])
    total = float(eigenvalues.sum())
    ratios = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    return CovariancePCA(mean, loadings, eigenvalues, ratios)


def build_daily_curve_changes(curve_history: pd.DataFrame) -> pd.DataFrame:
    """Return consecutive observed-date changes on the annual grid, in bp."""

    required = {"date", "maturity_years", "zero_rate_decimal"}
    missing = sorted(required.difference(curve_history.columns))
    if missing:
        raise ValueError(f"Curve history is missing columns: {missing}.")
    annual = curve_history.loc[
        curve_history["maturity_years"].isin(NODE_GRID),
        ["date", "maturity_years", "zero_rate_decimal"],
    ].copy()
    duplicates = annual.duplicated(["date", "maturity_years"], keep=False)
    if duplicates.any():
        raise ValueError("Curve history has duplicate date/maturity observations.")
    level = annual.pivot(
        index="date", columns="maturity_years", values="zero_rate_decimal"
    ).reindex(columns=NODE_GRID)
    level.index = pd.to_datetime(level.index).normalize()
    level = level.sort_index().dropna(how="any")
    changes = level.diff().dropna(how="any") * 10_000.0
    changes.columns = [f"{int(term)}Y_change_bp" for term in NODE_GRID]
    changes.index.name = "change_date"
    return changes


def trailing_training_sample(
    daily_changes: pd.DataFrame,
    event_date: pd.Timestamp,
    max_observations: int = MAX_TRAINING_OBSERVATIONS,
) -> pd.DataFrame:
    """Select at most the latest ``max_observations`` strictly pre-event rows."""

    event = pd.Timestamp(event_date).normalize()
    return daily_changes.loc[daily_changes.index < event].tail(max_observations)


def _instrument_krd(
    event_rows: pd.DataFrame, initial_zero_rates_bp: np.ndarray
) -> dict[str, np.ndarray]:
    curve = YieldCurve(NODE_GRID, initial_zero_rates_bp / 10_000.0)
    vectors: dict[str, np.ndarray] = {}
    for maturity in (2.0, 5.0, 10.0):
        name = f"{int(maturity)}Y_payer"
        row = event_rows.loc[event_rows["instrument"].eq(name)].iloc[0]
        swap = OISSwap(
            float(row["notional_eur"]),
            float(row["fixed_coupon_bp"]) / 10_000.0,
            maturity,
            side="payer",
        )
        vectors[name] = key_rate_sensitivities(swap, curve, NODE_GRID)[
            "sensitivity_eur_per_bp"
        ].to_numpy(float)

    portfolio_row = event_rows.loc[
        event_rows["instrument"].eq("2Y_payer_10Y_receiver_dv01_neutral")
    ].iloc[0]
    receiver = OISSwap(
        float(portfolio_row["portfolio_10y_receiver_notional_eur"]),
        curve.par_swap_rate(10.0),
        10.0,
        side="receiver",
    )
    receiver_krd = key_rate_sensitivities(receiver, curve, NODE_GRID)[
        "sensitivity_eur_per_bp"
    ].to_numpy(float)
    vectors["2Y_payer_10Y_receiver_dv01_neutral"] = (
        vectors["2Y_payer"] + receiver_krd
    )
    return vectors


def _error_metrics(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    instruments = [*sorted(results["instrument"].unique()), "ALL"]
    for instrument in instruments:
        subset = results if instrument == "ALL" else results.loc[
            results["instrument"].eq(instrument)
        ]
        actual = subset["full_repricing_pnl_eur"].to_numpy(float)
        risk = subset["krd_l2_risk_scale_eur_per_bp"].to_numpy(float)
        for method, column in METHOD_COLUMNS.items():
            errors = subset[column].to_numpy(float) - actual
            normalized = errors / risk
            rows.append(
                {
                    "instrument": instrument,
                    "approximation": method,
                    "n_event_instrument_rows": len(subset),
                    "n_distinct_events": subset["event_date"].nunique(),
                    "rmse_eur": float(np.sqrt(np.mean(errors**2))),
                    "mae_eur": float(np.mean(np.abs(errors))),
                    "mean_error_eur": float(np.mean(errors)),
                    "max_absolute_error_eur": float(np.max(np.abs(errors))),
                    "normalized_rmse_bp_equivalent": float(
                        np.sqrt(np.mean(normalized**2))
                    ),
                    "normalized_mae_bp_equivalent": float(
                        np.mean(np.abs(normalized))
                    ),
                    "normalization": "absolute error / L2 norm of annual zero-node KRD vector",
                }
            )
    return pd.DataFrame(rows)


def bootstrap_squared_error_differences(
    results: pd.DataFrame,
    n_bootstrap: int = 2_000,
    seed: int = 20261122,
) -> pd.DataFrame:
    """Paired event-cluster bootstrap of MSE differences (method A minus B)."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    rng = np.random.default_rng(seed)
    methods = list(METHOD_COLUMNS)
    rows: list[dict[str, object]] = []
    instruments = [*sorted(results["instrument"].unique()), "ALL"]
    for instrument in instruments:
        subset = results if instrument == "ALL" else results.loc[
            results["instrument"].eq(instrument)
        ]
        event_dates = np.sort(subset["event_date"].unique())
        event_losses: dict[str, np.ndarray] = {}
        for method, column in METHOD_COLUMNS.items():
            temp = subset.assign(
                squared_error=(subset[column] - subset["full_repricing_pnl_eur"]) ** 2
            )
            event_losses[method] = (
                temp.groupby("event_date")["squared_error"]
                .mean()
                .reindex(event_dates)
                .to_numpy(float)
            )
        draws = rng.integers(0, len(event_dates), size=(n_bootstrap, len(event_dates)))
        for left_index, method_a in enumerate(methods):
            for method_b in methods[left_index + 1 :]:
                paired = event_losses[method_a] - event_losses[method_b]
                boot = paired[draws].mean(axis=1)
                rows.append(
                    {
                        "instrument": instrument,
                        "method_a": method_a,
                        "method_b": method_b,
                        "n_events": len(event_dates),
                        "mean_squared_error_difference_eur2": float(paired.mean()),
                        "ci_2_5pct_eur2": float(np.quantile(boot, 0.025)),
                        "ci_97_5pct_eur2": float(np.quantile(boot, 0.975)),
                        "n_bootstrap": n_bootstrap,
                        "seed": seed,
                        "interpretation": "negative values favor method_a",
                    }
                )
    return pd.DataFrame(rows)


def _plot_loadings(loadings: pd.DataFrame, variance: pd.DataFrame, output: Path) -> None:
    latest = loadings["event_date"].max()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = ["#22577a", "#38a3a5", "#d97706"]
    for component, color in zip((1, 2, 3), colors, strict=True):
        subset = loadings.loc[loadings["component"].eq(component)]
        summary = subset.groupby("maturity_years")["loading"].agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        axes[0].plot(summary.index, summary["median"], marker="o", color=color,
                     label=f"PC{component}")
        axes[0].fill_between(summary.index, summary["q25"], summary["q75"],
                             color=color, alpha=0.14)
        reference = subset.loc[subset["event_date"].eq(latest)]
        ratio = variance.loc[
            variance["event_date"].eq(latest) & variance["component"].eq(component),
            "explained_variance_ratio",
        ].iloc[0]
        axes[1].plot(reference["maturity_years"], reference["loading"], marker="o",
                     color=color, label=f"PC{component} ({ratio:.1%})")
    axes[0].set_title("Across-event median (band: interquartile range)")
    axes[1].set_title(f"Latest eligible event: {latest:%Y-%m-%d}")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xlabel("Maturity (years)")
        axis.set_xticks(NODE_GRID)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Loading (dimensionless)")
    figure.suptitle("Trailing historical ECB AAA zero-curve PCA loadings")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _methodology(
    eligibility: pd.DataFrame, results: pd.DataFrame, metrics: pd.DataFrame
) -> str:
    included = eligibility.loc[eligibility["included"]]
    dates = pd.to_datetime(included["event_date"])
    headline = metrics.loc[
        metrics["instrument"].eq("ALL")
        & metrics["approximation"].isin(["parallel_zero_only", "pca_1", "pca_2", "pca_3", "full_krd"])
    ]
    metric_lines = "\n".join(
        f"- `{row.approximation}`: RMSE EUR {row.rmse_eur:,.2f}; MAE EUR {row.mae_eur:,.2f}; "
        f"normalized RMSE {row.normalized_rmse_bp_equivalent:.3f} bp-equivalent."
        for row in headline.itertuples()
    )
    return f"""# PCA-based swap P&L attribution

## Sample and information set

The input event sample is the existing exact-date audit: a public ECB AAA curve and complete EA-MPD annual 1Y–10Y par-rate shocks are required. Of {len(eligibility)} eligible pricing events, **{len(included)}** pass the PCA history rule, from **{dates.min().date()}** to **{dates.max().date()}**. Each event uses at most the latest {MAX_TRAINING_OBSERVATIONS} complete public-curve daily changes dated strictly before the event, with a minimum of {MIN_TRAINING_OBSERVATIONS}. The exported eligibility table records the estimation dates and any exclusion. Event observations are never included in their own PCA, no future data are used, and missing changes are not imputed.

## PCA definition and units

Continuously compounded annual-node zero rates are differenced between consecutive available ECB curve dates and converted to basis points. PCA is fitted to the **centered, unstandardized covariance matrix** of the 1Y–10Y changes. Loadings are dimensionless, scores and reconstructed shocks are in bp, and eigenvalues are in bp squared. PC1 is oriented to have a positive mean loading, PC2 a positive loading/maturity correlation, and PC3 a positive belly-minus-ends loading. These deterministic sign choices change neither reconstruction nor P&L.

For event shock `x`, the k-factor reconstruction is `historical mean + V_k V_k' (x - historical mean)`. The exercise projects the **observed** event movement onto a historical factor space. It is an attribution/approximation, not an ex-ante forecast of the event and not an independently identified monetary-policy shock.

## P&L attribution

Each reconstructed zero shock is multiplied by the event-specific annual zero-node KRD vector (EUR per bp). The component table separates the historical-mean P&L, PC1, PC2, PC3, and the residual after three components; together they reconstruct full-KRD P&L. PCA-1/2/3, the existing arithmetic-mean parallel approximation, full annual-node KRD, and nonlinear full repricing are compared on exactly the same dated event/instrument rows, including the dynamically parallel-DV01-neutral 2Y payer/10Y receiver portfolio.

Across all common rows:

{metric_lines}

Normalized errors divide each absolute EUR pricing error by the L2 norm of that row's ten-node KRD vector (EUR per bp), yielding a bp-equivalent error relative to the instrument's local zero-curve risk. It is a comparability scale, not a volatility-adjusted performance measure.

## Bootstrap and limitations

The paired bootstrap resamples event dates with replacement and retains every instrument for a selected date. Confidence intervals cover mean differences in squared errors, `method_a - method_b`; negative values favor method A. This respects the common event sample and within-event cross-instrument dependence, but it does not correct for possible dependence across nearby policy dates.

The initial curve remains the public ECB AAA euro-area central-government **sovereign proxy**, not an actual EUR OIS/€STR discount curve. EA-MPD par shocks were re-bootstrapped to zero shocks in the preceding pricing stage. Results are stylised counterfactual valuations, not historical dealer marks. Existing monetary-policy factor regressions and prior pricing outputs are not modified.
"""


def run(
    curve_path: Path,
    pricing_results_dir: Path,
    output_dir: Path,
    n_bootstrap: int = 2_000,
) -> None:
    curve_history = load_cached_aaa_curve(curve_path)
    daily_changes = build_daily_curve_changes(curve_history)
    event_pnl = pd.read_csv(
        pricing_results_dir / "event_level_pnl_comparison.csv", parse_dates=["event_date"]
    )
    zero_nodes = pd.read_csv(
        pricing_results_dir / "event_zero_curve_shocks.csv", parse_dates=["event_date"]
    )
    event_dates = sorted(zero_nodes["event_date"].unique())

    eligibility_rows: list[dict[str, object]] = []
    pnl_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []
    variance_rows: list[dict[str, object]] = []
    max_reconstruction_error = 0.0
    max_krd_consistency_error = 0.0
    max_pnl_decomposition_error = 0.0
    max_orthonormality_error = 0.0

    for raw_event_date in event_dates:
        event_date = pd.Timestamp(raw_event_date).normalize()
        training = trailing_training_sample(daily_changes, event_date)
        included = len(training) >= MIN_TRAINING_OBSERVATIONS
        eligibility_rows.append(
            {
                "event_date": event_date,
                "included": included,
                "exclusion_reason": "" if included else "fewer than 250 strictly pre-event changes",
                "training_observations": len(training),
                "training_start_date": training.index.min() if len(training) else pd.NaT,
                "training_end_date": training.index.max() if len(training) else pd.NaT,
                "training_end_strictly_before_event": bool(
                    len(training) and training.index.max() < event_date
                ),
                "maximum_trailing_observations": MAX_TRAINING_OBSERVATIONS,
                "minimum_required_observations": MIN_TRAINING_OBSERVATIONS,
                "missing_observations_imputed": False,
                "future_observations_used": False,
            }
        )
        if not included:
            continue

        model = fit_covariance_pca(training.to_numpy(float))
        identity_error = np.max(
            np.abs(model.loadings.T @ model.loadings - np.eye(len(NODE_GRID)))
        )
        max_orthonormality_error = max(max_orthonormality_error, float(identity_error))
        nodes = zero_nodes.loc[zero_nodes["event_date"].eq(event_date)].sort_values(
            "maturity_years"
        )
        if tuple(nodes["maturity_years"].to_numpy(float)) != NODE_GRID:
            raise ValueError(f"Event {event_date.date()} does not have the annual node grid.")
        shock = nodes["zero_rate_change_bp"].to_numpy(float)
        initial = nodes["initial_zero_rate_bp"].to_numpy(float)
        scores = model.scores(shock)
        all_reconstruction = model.reconstruct(shock, len(NODE_GRID))
        max_reconstruction_error = max(
            max_reconstruction_error, float(np.max(np.abs(all_reconstruction - shock)))
        )

        for component in range(len(NODE_GRID)):
            variance_rows.append(
                {
                    "event_date": event_date,
                    "training_start_date": training.index.min(),
                    "training_end_date": training.index.max(),
                    "training_observations": len(training),
                    "component": component + 1,
                    "eigenvalue_bp2": model.eigenvalues_bp2[component],
                    "explained_variance_ratio": model.explained_variance_ratio[component],
                    "cumulative_explained_variance_ratio": model.explained_variance_ratio[: component + 1].sum(),
                }
            )
            for maturity_index, maturity in enumerate(NODE_GRID):
                loading_rows.append(
                    {
                        "event_date": event_date,
                        "training_start_date": training.index.min(),
                        "training_end_date": training.index.max(),
                        "component": component + 1,
                        "maturity_years": maturity,
                        "loading": model.loadings[maturity_index, component],
                    }
                )

        event_rows = event_pnl.loc[event_pnl["event_date"].eq(event_date)].copy()
        krd_vectors = _instrument_krd(event_rows, initial)
        component_shocks = [
            model.loadings[:, index] * scores[index] for index in range(3)
        ]
        reconstruction = {
            count: model.reconstruct(shock, count) for count in (1, 2, 3)
        }
        residual = shock - reconstruction[3]

        for row in event_rows.itertuples(index=False):
            instrument = row.instrument
            krd = krd_vectors[instrument]
            baseline_krd = float(row.zero_krd_approximation_eur)
            computed_krd = float(np.dot(krd, shock))
            max_krd_consistency_error = max(
                max_krd_consistency_error, abs(computed_krd - baseline_krd)
            )
            approximations = {
                count: float(np.dot(krd, reconstruction[count])) for count in (1, 2, 3)
            }
            risk_scale = float(np.linalg.norm(krd))
            result = {
                "event_date": event_date,
                "curve_date_used": row.curve_date_used,
                "date_alignment_days": row.date_alignment_days,
                "instrument": instrument,
                "full_repricing_pnl_eur": float(row.full_repricing_pnl_eur),
                "parallel_only_approximation_eur": float(row.parallel_only_approximation_eur),
                "zero_krd_approximation_eur": baseline_krd,
                "pca_1_approximation_eur": approximations[1],
                "pca_2_approximation_eur": approximations[2],
                "pca_3_approximation_eur": approximations[3],
                "krd_l2_risk_scale_eur_per_bp": risk_scale,
                "training_observations": len(training),
                "training_start_date": training.index.min(),
                "training_end_date": training.index.max(),
                "future_observations_used": False,
                "shock_imputation_used": False,
            }
            for method, column in METHOD_COLUMNS.items():
                error = float(result[column]) - float(row.full_repricing_pnl_eur)
                result[f"{method}_error_eur"] = error
                result[f"{method}_normalized_error_bp_equivalent"] = error / risk_scale
            pnl_rows.append(result)

            mean_pnl = float(np.dot(krd, model.mean_bp))
            pc_pnl = [float(np.dot(krd, value)) for value in component_shocks]
            residual_pnl = float(np.dot(krd, residual))
            reconstructed = mean_pnl + sum(pc_pnl) + residual_pnl
            max_pnl_decomposition_error = max(
                max_pnl_decomposition_error, abs(reconstructed - baseline_krd)
            )
            contribution_rows.append(
                {
                    "event_date": event_date,
                    "instrument": instrument,
                    "historical_mean_contribution_eur": mean_pnl,
                    "pc1_contribution_eur": pc_pnl[0],
                    "pc2_contribution_eur": pc_pnl[1],
                    "pc3_contribution_eur": pc_pnl[2],
                    "residual_after_pc3_eur": residual_pnl,
                    "sum_contributions_eur": reconstructed,
                    "full_krd_approximation_eur": baseline_krd,
                    "reconstruction_difference_eur": reconstructed - baseline_krd,
                }
            )

    eligibility = pd.DataFrame(eligibility_rows)
    results = pd.DataFrame(pnl_rows)
    contributions = pd.DataFrame(contribution_rows)
    loadings = pd.DataFrame(loading_rows)
    variance = pd.DataFrame(variance_rows)
    metrics = _error_metrics(results)
    bootstrap = bootstrap_squared_error_differences(results, n_bootstrap=n_bootstrap)

    included_eligibility = eligibility.loc[eligibility["included"]]
    loading_grid = {
        component: loadings.loc[loadings["component"].eq(component)].pivot(
            index="event_date", columns="maturity_years", values="loading"
        )
        for component in (1, 2, 3)
    }
    pc2_correlations = loading_grid[2].apply(
        lambda row: np.corrcoef(row.to_numpy(float), np.asarray(NODE_GRID))[0, 1],
        axis=1,
    )
    pc3_curvature = loading_grid[3][[4.0, 5.0, 6.0, 7.0]].mean(axis=1) - loading_grid[3][
        [1.0, 2.0, 9.0, 10.0]
    ].mean(axis=1)
    validation = pd.DataFrame(
        [
            {"check": "minimum training observations", "value": included_eligibility["training_observations"].min(), "tolerance": f">={MIN_TRAINING_OBSERVATIONS}", "passed": included_eligibility["training_observations"].min() >= MIN_TRAINING_OBSERVATIONS},
            {"check": "strictly pre-event training end", "value": int(included_eligibility["training_end_strictly_before_event"].all()), "tolerance": "true", "passed": bool(included_eligibility["training_end_strictly_before_event"].all())},
            {"check": "loading orthonormality max abs error", "value": max_orthonormality_error, "tolerance": "<1e-10", "passed": max_orthonormality_error < 1e-10},
            {"check": "all-component reconstruction max abs bp error", "value": max_reconstruction_error, "tolerance": "<1e-10 bp", "passed": max_reconstruction_error < 1e-10},
            {"check": "recomputed versus existing KRD P&L max abs error", "value": max_krd_consistency_error, "tolerance": "<0.01 EUR", "passed": max_krd_consistency_error < 0.01},
            {"check": "mean plus PC1-3 plus residual P&L max abs error", "value": max_pnl_decomposition_error, "tolerance": "<1e-8 EUR", "passed": max_pnl_decomposition_error < 1e-8},
            {"check": "PC1 positive mean sign", "value": loadings.loc[loadings["component"].eq(1)].groupby("event_date")["loading"].mean().min(), "tolerance": ">0", "passed": loadings.loc[loadings["component"].eq(1)].groupby("event_date")["loading"].mean().min() > 0},
            {"check": "PC2 positive loading-maturity correlation sign", "value": pc2_correlations.min(), "tolerance": ">0", "passed": pc2_correlations.min() > 0},
            {"check": "PC3 positive belly-minus-ends curvature sign", "value": pc3_curvature.min(), "tolerance": ">0", "passed": pc3_curvature.min() > 0},
            {"check": "units", "value": "shock/score=bp; loading=dimensionless; eigenvalue=bp^2; KRD=EUR/bp", "tolerance": "documented", "passed": True},
        ]
    )
    if not validation["passed"].all():
        failed = validation.loc[~validation["passed"], "check"].tolist()
        raise RuntimeError(f"PCA validation failed: {failed}")

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_changes.reset_index().to_csv(output_dir / "daily_aaa_zero_curve_changes.csv", index=False, date_format="%Y-%m-%d")
    eligibility.to_csv(output_dir / "pca_event_eligibility.csv", index=False, date_format="%Y-%m-%d")
    results.to_csv(output_dir / "event_level_pca_pnl.csv", index=False, date_format="%Y-%m-%d")
    contributions.to_csv(output_dir / "pca_component_pnl_contributions.csv", index=False, date_format="%Y-%m-%d")
    variance.to_csv(output_dir / "pca_variance_explained.csv", index=False, date_format="%Y-%m-%d")
    loadings.to_csv(output_dir / "pca_loadings_by_event.csv", index=False, date_format="%Y-%m-%d")
    metrics.to_csv(output_dir / "pca_pnl_error_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_squared_error_differences.csv", index=False)
    validation.to_csv(output_dir / "pca_validation.csv", index=False)
    _plot_loadings(loadings, variance, output_dir / "pca_loadings.png")
    (output_dir / "methodology.md").write_text(
        _methodology(eligibility, results, metrics), encoding="utf-8"
    )
    metadata = {
        "curve_source": str(curve_path),
        "pricing_results_source": str(pricing_results_dir),
        "output_directory": str(output_dir),
        "node_grid_years": NODE_GRID,
        "pca": "centered unstandardized covariance PCA",
        "training_window_maximum_changes": MAX_TRAINING_OBSERVATIONS,
        "minimum_training_changes": MIN_TRAINING_OBSERVATIONS,
        "eligible_input_events": len(eligibility),
        "included_pca_events": int(eligibility["included"].sum()),
        "included_event_instrument_rows": len(results),
        "bootstrap_resamples": n_bootstrap,
        "bootstrap_seed": 20261122,
        "units": {
            "zero_changes_and_scores": "basis points",
            "loadings": "dimensionless",
            "eigenvalues": "basis points squared",
            "key_rate_sensitivities": "EUR per basis point",
            "pnl": "EUR",
        },
        "sovereign_proxy_not_ois_curve": True,
        "pca_is_observed_shock_attribution_not_prediction": True,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve",
        type=Path,
        default=Path("data/external/ecb_aaa_curve/aaa_spot_rates_3m_10y.csv"),
    )
    parser.add_argument("--pricing-results", type=Path, default=Path("results/swap_pricing"))
    parser.add_argument("--output", type=Path, default=Path("results/pca_pnl"))
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    args = parser.parse_args()
    run(args.curve, args.pricing_results, args.output, args.bootstrap_resamples)


if __name__ == "__main__":
    main()
