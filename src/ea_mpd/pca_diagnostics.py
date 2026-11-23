"""Audit saved PCA P&L results and prepare a factor-attribution design."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pca_pnl import fit_covariance_pca, trailing_training_sample
from .swap_pricing import YieldCurve, bootstrap_par_swap_curve
from .swap_risk_experiments import NODE_GRID


INSTRUMENTS = (
    "2Y_payer",
    "5Y_payer",
    "10Y_payer",
    "2Y_payer_10Y_receiver_dv01_neutral",
)
METHODS = ("parallel_zero_only", "pca_1", "pca_2", "pca_3", "full_krd")
BOOTSTRAP_PAIRS = (
    ("pca_1", "pca_2"),
    ("pca_2", "pca_3"),
    ("pca_3", "full_krd"),
    ("parallel_zero_only", "pca_3"),
)
FACTORS = ("target", "timing", "forward_guidance", "qe")
FACTOR_WINDOWS = {
    "target": "Press Release Window",
    "timing": "Press Conference Window",
    "forward_guidance": "Press Conference Window",
    "qe": "Press Conference Window",
}
FACTOR_NORMALIZATIONS = {
    "target": "unit effect on 1M OIS",
    "timing": "unit effect on 6M OIS",
    "forward_guidance": "unit effect on 2Y OIS",
    "qe": "unit effect on 10Y OIS",
}
WINDOW_CHANGE_DATE = pd.Timestamp("2022-07-01")
REFERENCE_DATE = pd.Timestamp("2025-10-30")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_error_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select the common four-instrument, five-method PCA audit table."""

    required = {
        "instrument",
        "approximation",
        "n_distinct_events",
        "rmse_eur",
        "mae_eur",
        "normalized_rmse_bp_equivalent",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"PCA metrics are missing columns: {missing}.")
    selected = metrics.loc[
        metrics["instrument"].isin(INSTRUMENTS)
        & metrics["approximation"].isin(METHODS),
        [
            "instrument",
            "approximation",
            "n_distinct_events",
            "rmse_eur",
            "mae_eur",
            "normalized_rmse_bp_equivalent",
        ],
    ].copy()
    selected["instrument"] = pd.Categorical(
        selected["instrument"], categories=INSTRUMENTS, ordered=True
    )
    selected["approximation"] = pd.Categorical(
        selected["approximation"], categories=METHODS, ordered=True
    )
    selected = selected.sort_values(["instrument", "approximation"]).reset_index(
        drop=True
    )
    expected = len(INSTRUMENTS) * len(METHODS)
    if len(selected) != expected or not selected["n_distinct_events"].eq(128).all():
        raise ValueError("PCA method comparison is not the expected 4 x 5 x 128 panel.")
    return selected


def summarize_explained_variance(variance: pd.DataFrame) -> pd.DataFrame:
    """Summarize rolling explained-variance distributions across events."""

    required = {
        "event_date",
        "component",
        "explained_variance_ratio",
        "cumulative_explained_variance_ratio",
    }
    missing = sorted(required.difference(variance.columns))
    if missing:
        raise ValueError(f"Variance table is missing columns: {missing}.")
    if variance["event_date"].nunique() != 128:
        raise ValueError("Explained-variance audit requires exactly 128 events.")
    rows: list[dict[str, object]] = []
    for component, group in variance.groupby("component", sort=True):
        explained = group["explained_variance_ratio"]
        cumulative = group["cumulative_explained_variance_ratio"]
        rows.append(
            {
                "component": int(component),
                "n_events": group["event_date"].nunique(),
                "explained_mean": explained.mean(),
                "explained_median": explained.median(),
                "explained_p05": explained.quantile(0.05),
                "explained_p25": explained.quantile(0.25),
                "explained_p75": explained.quantile(0.75),
                "explained_p95": explained.quantile(0.95),
                "explained_min": explained.min(),
                "explained_max": explained.max(),
                "cumulative_mean": cumulative.mean(),
                "cumulative_median": cumulative.median(),
                "cumulative_p05": cumulative.quantile(0.05),
                "cumulative_p95": cumulative.quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def select_bootstrap_comparisons(bootstrap: pd.DataFrame) -> pd.DataFrame:
    """Select requested paired-bootstrap comparisons and interpret their CIs."""

    keys = pd.MultiIndex.from_tuples(BOOTSTRAP_PAIRS, names=["method_a", "method_b"])
    selected = bootstrap.set_index(["method_a", "method_b"]).loc[keys].reset_index()
    selected = selected.loc[
        selected["instrument"].isin((*INSTRUMENTS, "ALL"))
    ].copy()
    lower = selected["ci_2_5pct_eur2"]
    upper = selected["ci_97_5pct_eur2"]
    selected["ci_excludes_zero"] = (lower > 0) | (upper < 0)
    selected["favored_method_95pct"] = np.select(
        [upper < 0, lower > 0],
        [selected["method_a"], selected["method_b"]],
        default="inconclusive",
    )
    selected["difference_definition"] = "mean squared error(method_a) - mean squared error(method_b)"
    return selected.sort_values(["method_a", "method_b", "instrument"]).reset_index(
        drop=True
    )


def zero_shock_reconstruction_test(
    daily_changes: pd.DataFrame, eligibility: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test affine PCA reconstruction at the economically neutral zero shock."""

    rows: list[dict[str, object]] = []
    zero = np.zeros(len(NODE_GRID))
    for event_date in pd.to_datetime(
        eligibility.loc[eligibility["included"], "event_date"]
    ):
        training = trailing_training_sample(daily_changes, event_date)
        model = fit_covariance_pca(training.to_numpy(float))
        for components in (1, 2, 3, len(NODE_GRID)):
            affine = model.reconstruct(zero, components)
            linear = (
                model.loadings[:, :components]
                @ model.loadings[:, :components].T
                @ zero
            )
            rows.append(
                {
                    "event_date": event_date,
                    "components": components,
                    "training_observations": len(training),
                    "affine_zero_reconstruction_mean_bp": float(np.mean(affine)),
                    "affine_zero_reconstruction_rms_bp": float(
                        np.sqrt(np.mean(affine**2))
                    ),
                    "affine_zero_reconstruction_l2_bp": float(np.linalg.norm(affine)),
                    "affine_zero_reconstruction_max_abs_bp": float(
                        np.max(np.abs(affine))
                    ),
                    "origin_preserving_zero_reconstruction_max_abs_bp": float(
                        np.max(np.abs(linear))
                    ),
                    "affine_formula": "mean + V_k V_k' (zero - mean)",
                    "origin_preserving_formula": "V_k V_k' zero",
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for components, group in detail.groupby("components", sort=True):
        rms = group["affine_zero_reconstruction_rms_bp"]
        maximum = group["affine_zero_reconstruction_max_abs_bp"]
        summary_rows.append(
            {
                "components": int(components),
                "n_events": group["event_date"].nunique(),
                "affine_rms_bp_mean": rms.mean(),
                "affine_rms_bp_median": rms.median(),
                "affine_rms_bp_p95": rms.quantile(0.95),
                "affine_rms_bp_max": rms.max(),
                "affine_max_abs_bp_mean": maximum.mean(),
                "affine_max_abs_bp_p95": maximum.quantile(0.95),
                "affine_max_abs_bp_max": maximum.max(),
                "origin_preserving_max_abs_bp": group[
                    "origin_preserving_zero_reconstruction_max_abs_bp"
                ].max(),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def _window_panel(ois: pd.DataFrame, eligible_dates: set[pd.Timestamp]) -> pd.DataFrame:
    annual = ois.loc[
        ois["maturity_years"].isin(NODE_GRID)
        & ois["event_date"].isin(eligible_dates),
        ["event_date", "maturity_years", "window", "surprise_bp"],
    ]
    duplicates = annual.duplicated(["event_date", "maturity_years", "window"], False)
    if duplicates.any():
        raise ValueError("Duplicate annual OIS observations in window audit.")
    panel = annual.pivot(
        index=["event_date", "maturity_years"], columns="window", values="surprise_bp"
    ).reset_index()
    expected = {
        "Monetary Event Window",
        "Press Release Window",
        "Press Conference Window",
    }
    if not expected.issubset(panel.columns):
        raise ValueError("Required EA-MPD windows are absent from OIS surprises.")
    panel["release_plus_conference_bp"] = (
        panel["Press Release Window"] + panel["Press Conference Window"]
    )
    panel["monetary_minus_component_windows_bp"] = (
        panel["Monetary Event Window"] - panel["release_plus_conference_bp"]
    )
    panel["window_regime"] = np.where(
        panel["event_date"] < WINDOW_CHANGE_DATE,
        "through 2022-06 (non-shared endpoints)",
        "from 2022-07 (shared release/conference endpoint)",
    )
    return panel


def audit_factor_coverage(
    eligibility: pd.DataFrame,
    merged: pd.DataFrame,
    ois: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit published-factor coverage and window consistency on PCA events."""

    eligible = eligibility.loc[eligibility["included"], ["event_date"]].copy()
    eligible["event_date"] = pd.to_datetime(eligible["event_date"])
    factor_columns = ["event_date", *FACTORS, "factor_vintage"]
    event_audit = eligible.merge(
        merged[factor_columns], on="event_date", how="left", validate="one_to_one"
    )
    window_panel = _window_panel(ois, set(event_audit["event_date"]))
    completeness = (
        window_panel.groupby("event_date")[[
            "Press Release Window",
            "Press Conference Window",
            "Monetary Event Window",
        ]]
        .agg(lambda values: bool(values.notna().all() and len(values) == len(NODE_GRID)))
        .rename(
            columns={
                "Press Release Window": "release_annual_1y_10y_complete",
                "Press Conference Window": "conference_annual_1y_10y_complete",
                "Monetary Event Window": "monetary_annual_1y_10y_complete",
            }
        )
    )
    additivity = window_panel.groupby("event_date").agg(
        window_component_gap_max_abs_bp=(
            "monetary_minus_component_windows_bp",
            lambda values: values.abs().max(),
        ),
        window_component_gap_rms_bp=(
            "monetary_minus_component_windows_bp",
            lambda values: float(np.sqrt(np.mean(values**2))),
        ),
    )
    event_audit = event_audit.join(completeness, on="event_date").join(
        additivity, on="event_date"
    )
    for factor in FACTORS:
        event_audit[f"{factor}_available"] = event_audit[factor].notna()
    event_audit["all_four_factors_available"] = event_audit[
        [f"{factor}_available" for factor in FACTORS]
    ].all(axis=1)
    event_audit["window_regime"] = np.where(
        event_audit["event_date"] < WINDOW_CHANGE_DATE,
        "through 2022-06 (non-shared endpoints)",
        "from 2022-07 (shared release/conference endpoint)",
    )
    event_audit["component_windows_add_to_monetary_window"] = (
        event_audit["window_component_gap_max_abs_bp"] < 1.0e-8
    )
    event_audit["proposed_complete_factor_sample"] = (
        event_audit["all_four_factors_available"]
        & event_audit["release_annual_1y_10y_complete"]
        & event_audit["conference_annual_1y_10y_complete"]
        & event_audit["monetary_annual_1y_10y_complete"]
    )
    event_audit["exclusion_reason"] = np.where(
        event_audit["proposed_complete_factor_sample"],
        "",
        np.where(~event_audit["qe_available"], "QE factor missing", "other validation failure"),
    )
    event_audit = event_audit.drop(columns=list(FACTORS))

    coverage_rows: list[dict[str, object]] = []
    for factor in FACTORS:
        all_values = merged.loc[merged[factor].notna(), "event_date"]
        selected_values = event_audit.loc[event_audit[f"{factor}_available"], "event_date"]
        coverage_rows.append(
            {
                "factor": factor,
                "event_window": FACTOR_WINDOWS[factor],
                "normalization": FACTOR_NORMALIZATIONS[factor],
                "published_nonmissing_events": len(all_values),
                "published_start": all_values.min(),
                "published_end": all_values.max(),
                "available_among_128_pca_events": len(selected_values),
                "pca_event_start": selected_values.min(),
                "pca_event_end": selected_values.max(),
                "missing_among_128_pca_events": 128 - len(selected_values),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.loc[len(coverage)] = {
        "factor": "all_four_complete",
        "event_window": "release and conference kept separate",
        "normalization": "not applicable",
        "published_nonmissing_events": merged[list(FACTORS)].notna().all(axis=1).sum(),
        "published_start": merged.loc[merged[list(FACTORS)].notna().all(axis=1), "event_date"].min(),
        "published_end": merged.loc[merged[list(FACTORS)].notna().all(axis=1), "event_date"].max(),
        "available_among_128_pca_events": event_audit["all_four_factors_available"].sum(),
        "pca_event_start": event_audit.loc[event_audit["all_four_factors_available"], "event_date"].min(),
        "pca_event_end": event_audit.loc[event_audit["all_four_factors_available"], "event_date"].max(),
        "missing_among_128_pca_events": (~event_audit["all_four_factors_available"]).sum(),
    }

    gap_summary = (
        window_panel.groupby("window_regime")
        .agg(
            n_events=("event_date", "nunique"),
            n_event_maturity_rows=("event_date", "size"),
            mean_abs_gap_bp=(
                "monetary_minus_component_windows_bp",
                lambda values: values.abs().mean(),
            ),
            rms_gap_bp=(
                "monetary_minus_component_windows_bp",
                lambda values: float(np.sqrt(np.mean(values**2))),
            ),
            max_abs_gap_bp=(
                "monetary_minus_component_windows_bp",
                lambda values: values.abs().max(),
            ),
            rows_exceeding_1e_8_bp=(
                "monetary_minus_component_windows_bp",
                lambda values: int((values.abs() > 1.0e-8).sum()),
            ),
        )
        .reset_index()
    )
    return event_audit, coverage, window_panel, gap_summary


def par_to_zero_conversion_validation(
    coefficients: pd.DataFrame, zero_nodes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate unit-factor par-to-zero mapping without attributing event P&L."""

    annual_labels = [f"{int(term)}Y" for term in NODE_GRID]
    annual = coefficients.loc[
        coefficients["factor"].isin(FACTORS)
        & coefficients["maturity"].isin(annual_labels)
    ].copy()
    pivot = annual.pivot(index="maturity", columns="factor", values="coefficient").reindex(
        annual_labels
    )
    if pivot.isna().any().any() or tuple(pivot.columns) != tuple(sorted(FACTORS)):
        missing = pivot.isna().sum().to_dict()
        raise ValueError(f"Annual factor loading grid is incomplete: {missing}.")

    nodes = zero_nodes.loc[zero_nodes["event_date"].eq(REFERENCE_DATE)].sort_values(
        "maturity_years"
    )
    if tuple(nodes["maturity_years"].to_numpy(float)) != NODE_GRID:
        raise ValueError("Reference event lacks the annual 1Y--10Y zero grid.")
    base_curve = YieldCurve(
        np.asarray(NODE_GRID), nodes["initial_zero_rate_bp"].to_numpy(float) / 10_000.0
    )
    base_par = np.asarray([base_curve.par_swap_rate(term) for term in NODE_GRID])
    jacobian = np.empty((len(NODE_GRID), len(NODE_GRID)))
    for quote_index in range(len(NODE_GRID)):
        quote_bump = np.zeros(len(NODE_GRID))
        quote_bump[quote_index] = 1.0e-4
        up = bootstrap_par_swap_curve(NODE_GRID, base_par + quote_bump)
        down = bootstrap_par_swap_curve(NODE_GRID, base_par - quote_bump)
        jacobian[:, quote_index] = (
            (up.zero_rates_cc - down.zero_rates_cc) * 10_000.0 / 2.0
        )

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for factor in FACTORS:
        par_shock_bp = pivot[factor].to_numpy(float)
        exact_curve = bootstrap_par_swap_curve(
            NODE_GRID, base_par + par_shock_bp * 1.0e-4
        )
        exact_zero_bp = (
            exact_curve.zero_rates_cc - base_curve.zero_rates_cc
        ) * 10_000.0
        linear_zero_bp = jacobian @ par_shock_bp
        reproduced_par_bp = np.asarray(
            [exact_curve.par_swap_rate(term) for term in NODE_GRID]
        ) * 10_000.0
        target_par_bp = base_par * 10_000.0 + par_shock_bp
        for index, maturity in enumerate(NODE_GRID):
            detail_rows.append(
                {
                    "reference_date": REFERENCE_DATE,
                    "factor": factor,
                    "factor_units": 1.0,
                    "factor_window": FACTOR_WINDOWS[factor],
                    "maturity_years": maturity,
                    "fitted_par_ois_shock_bp": par_shock_bp[index],
                    "exact_rebootstrapped_zero_shock_bp": exact_zero_bp[index],
                    "linear_jacobian_zero_shock_bp": linear_zero_bp[index],
                    "linear_minus_exact_zero_shock_bp": linear_zero_bp[index]
                    - exact_zero_bp[index],
                    "naive_par_equals_zero_error_bp": par_shock_bp[index]
                    - exact_zero_bp[index],
                    "reproduced_shocked_par_rate_bp": reproduced_par_bp[index],
                    "target_shocked_par_rate_bp": target_par_bp[index],
                    "par_reproduction_error_bp": reproduced_par_bp[index]
                    - target_par_bp[index],
                }
            )
        summary_rows.append(
            {
                "reference_date": REFERENCE_DATE,
                "factor": factor,
                "max_abs_par_reproduction_error_bp": float(
                    np.max(np.abs(reproduced_par_bp - target_par_bp))
                ),
                "max_abs_linear_vs_exact_zero_error_bp": float(
                    np.max(np.abs(linear_zero_bp - exact_zero_bp))
                ),
                "max_abs_naive_par_equals_zero_error_bp": float(
                    np.max(np.abs(par_shock_bp - exact_zero_bp))
                ),
                "conversion_status": "validated for unit factor loading vector",
            }
        )
    jacobian_rows = [
        {
            "reference_date": REFERENCE_DATE,
            "zero_maturity_years": NODE_GRID[zero_index],
            "par_quote_maturity_years": NODE_GRID[quote_index],
            "zero_bp_per_1bp_par_quote": jacobian[zero_index, quote_index],
        }
        for zero_index in range(len(NODE_GRID))
        for quote_index in range(len(NODE_GRID))
    ]
    return (
        pd.DataFrame(detail_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(jacobian_rows),
    )


def factor_attribution_design() -> pd.DataFrame:
    """Return the gated design; this module deliberately does not execute it."""

    rows = [
        (1, "Freeze sources", "Use the 2025-10-30 common factor vintage and recorded hashes; never mix release and conference vintages.", "validated"),
        (2, "Choose common sample", "Require exact-date base curve, annual 1Y-10Y pricing grid, Target, Timing, Forward Guidance and QE; no imputation.", "99 events validated"),
        (3, "Keep windows separate", "Target multiplies press-release loadings; Timing, Forward Guidance and QE jointly multiply press-conference loadings.", "validated"),
        (4, "Build fitted par vectors", "At each annual maturity, fitted release shock = beta_Target x Target; fitted conference shock = beta_Timing x Timing + beta_FG x FG + beta_QE x QE.", "design only"),
        (5, "Retain window residual", "Do not force release plus conference to equal the monetary-event window before July 2022; report the timing-gap and regression residual separately.", "validated limitation"),
        (6, "Convert par shocks to zero shocks", "At each event, add fitted par shocks to the base par curve and rebootstrap. For additive factor contributions use the event-specific par-to-zero Jacobian; report nonlinear rebootstrap remainder.", "mechanics validated on unit vectors"),
        (7, "Apply swap risk", "Multiply each factor zero vector by the existing event-specific zero-node KRD. Compare the sum plus residual with full KRD and full repricing.", "not executed"),
        (8, "Interpret cautiously", "Treat the result as a generated-factor decomposition because the rates used as outcomes helped construct and normalize the factors.", "required boundary"),
    ]
    return pd.DataFrame(rows, columns=["step", "stage", "method", "status"])


def _plot_metrics(metrics: pd.DataFrame, destination: Path) -> None:
    labels = {
        "parallel_zero_only": "Parallel",
        "pca_1": "PCA-1",
        "pca_2": "PCA-2",
        "pca_3": "PCA-3",
        "full_krd": "KRD",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(INSTRUMENTS))
    width = 0.15
    for method_index, method in enumerate(METHODS):
        subset = metrics.loc[metrics["approximation"].eq(method)].set_index("instrument")
        positions = x + (method_index - 2) * width
        axes[0].bar(positions, subset.loc[list(INSTRUMENTS), "rmse_eur"], width, label=labels[method])
        axes[1].bar(
            positions,
            subset.loc[list(INSTRUMENTS), "normalized_rmse_bp_equivalent"],
            width,
            label=labels[method],
        )
    short_names = ["2Y payer", "5Y payer", "10Y payer", "2Y/10Y portfolio"]
    for axis in axes:
        axis.set_xticks(x, short_names, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("Pricing-error RMSE")
    axes[0].set_ylabel("EUR")
    axes[1].set_title("Risk-normalized RMSE")
    axes[1].set_ylabel("bp-equivalent")
    axes[1].legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_variance(variance: pd.DataFrame, destination: Path) -> None:
    first = variance.loc[variance["component"].le(5)].copy()
    data = [
        first.loc[first["component"].eq(component), "explained_variance_ratio"] * 100
        for component in range(1, 6)
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5))
    axis.boxplot(data, tick_labels=[f"PC{component}" for component in range(1, 6)], showfliers=False)
    axis.set_ylabel("Explained variance (%)")
    axis.set_title("Trailing PCA explained variance across 128 ECB events")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_bootstrap(selected: pd.DataFrame, destination: Path) -> None:
    figures, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    display_instruments = (*INSTRUMENTS, "ALL")
    short = ["2Y", "5Y", "10Y", "2Y/10Y", "ALL"]
    for axis, pair in zip(axes.flat, BOOTSTRAP_PAIRS, strict=True):
        subset = selected.loc[
            selected["method_a"].eq(pair[0]) & selected["method_b"].eq(pair[1])
        ].set_index("instrument").loc[list(display_instruments)]
        point = subset["mean_squared_error_difference_eur2"].to_numpy(float) / 1.0e6
        lower = subset["ci_2_5pct_eur2"].to_numpy(float) / 1.0e6
        upper = subset["ci_97_5pct_eur2"].to_numpy(float) / 1.0e6
        y = np.arange(len(short))
        axis.errorbar(point, y, xerr=[point - lower, upper - point], fmt="o", capsize=3)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y, short)
        axis.set_title(f"{pair[0]} minus {pair[1]}")
        axis.set_xlabel("Difference in MSE (EUR² millions)")
        axis.grid(axis="x", alpha=0.25)
    figures.suptitle("Paired event-bootstrap 95% confidence intervals")
    figures.tight_layout()
    figures.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figures)


def _plot_zero_test(detail: pd.DataFrame, destination: Path) -> None:
    groups = [
        detail.loc[detail["components"].eq(component), "affine_zero_reconstruction_rms_bp"]
        for component in (1, 2, 3, 10)
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5))
    axis.boxplot(groups, tick_labels=["PCA-1", "PCA-2", "PCA-3", "PCA-10"], showfliers=False)
    axis.set_ylabel("RMS reconstructed shock (bp)")
    axis.set_title("Affine PCA reconstruction when the input shock is zero")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_factor_conversion(detail: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    titles = {
        "target": "Target",
        "timing": "Timing",
        "forward_guidance": "Forward Guidance",
        "qe": "QE",
    }
    for axis, factor in zip(axes.flat, FACTORS, strict=True):
        subset = detail.loc[detail["factor"].eq(factor)]
        axis.plot(subset["maturity_years"], subset["fitted_par_ois_shock_bp"], marker="o", label="Fitted par shock")
        axis.plot(subset["maturity_years"], subset["exact_rebootstrapped_zero_shock_bp"], marker="s", label="Rebootstrapped zero shock")
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(f"One unit of {titles[factor]}")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Shock (bp)")
    axes[1, 0].set_ylabel("Shock (bp)")
    axes[1, 0].set_xlabel("Maturity (years)")
    axes[1, 1].set_xlabel("Maturity (years)")
    axes[0, 1].legend(fontsize=8)
    figure.suptitle("Par OIS factor vectors and implied zero-rate shocks")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _report(
    metrics: pd.DataFrame,
    variance_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    zero_summary: pd.DataFrame,
    factor_coverage: pd.DataFrame,
    gap_summary: pd.DataFrame,
    conversion_summary: pd.DataFrame,
) -> str:
    metric_lines = []
    for instrument in INSTRUMENTS:
        subset = metrics.loc[metrics["instrument"].eq(instrument)]
        entries = "; ".join(
            f"{row.approximation}: RMSE EUR {row.rmse_eur:,.0f}, MAE EUR {row.mae_eur:,.0f}, normalized RMSE {row.normalized_rmse_bp_equivalent:.3f} bp-equivalent"
            for row in subset.itertuples()
        )
        metric_lines.append(f"- **{instrument}**: {entries}.")
    variance_lines = []
    for component in (1, 2, 3):
        row = variance_summary.loc[variance_summary["component"].eq(component)].iloc[0]
        variance_lines.append(
            f"- PC{component}: mean {row['explained_mean']:.1%}, median {row['explained_median']:.1%}, 5th–95th percentile {row['explained_p05']:.1%}–{row['explained_p95']:.1%}; median cumulative {row['cumulative_median']:.1%}."
        )
    all_bootstrap = bootstrap.loc[bootstrap["instrument"].eq("ALL")]
    bootstrap_lines = [
        f"- `{row.method_a} - {row.method_b}`: difference EUR² {row.mean_squared_error_difference_eur2:,.0f}, 95% CI [{row.ci_2_5pct_eur2:,.0f}, {row.ci_97_5pct_eur2:,.0f}]; {row.favored_method_95pct} favored."
        for row in all_bootstrap.itertuples()
    ]
    instrument_bootstrap_lines = []
    for method_a, method_b in BOOTSTRAP_PAIRS:
        subset = bootstrap.loc[
            bootstrap["method_a"].eq(method_a)
            & bootstrap["method_b"].eq(method_b)
            & bootstrap["instrument"].isin(INSTRUMENTS)
        ]
        favored = subset.loc[subset["ci_excludes_zero"]].groupby(
            "favored_method_95pct"
        )["instrument"].apply(lambda values: ", ".join(values)).to_dict()
        inconclusive = subset.loc[~subset["ci_excludes_zero"], "instrument"].tolist()
        parts = [f"{method}: {names}" for method, names in favored.items()]
        if inconclusive:
            parts.append(f"inconclusive: {', '.join(inconclusive)}")
        instrument_bootstrap_lines.append(
            f"- `{method_a} - {method_b}` instrument-level CIs: " + "; ".join(parts) + "."
        )
    zero3 = zero_summary.loc[zero_summary["components"].eq(3)].iloc[0]
    full = zero_summary.loc[zero_summary["components"].eq(10)].iloc[0]
    complete = factor_coverage.loc[factor_coverage["factor"].eq("all_four_complete")].iloc[0]
    pre_gap = gap_summary.loc[gap_summary["window_regime"].str.startswith("through")].iloc[0]
    post_gap = gap_summary.loc[gap_summary["window_regime"].str.startswith("from")].iloc[0]
    max_par_error = conversion_summary["max_abs_par_reproduction_error_bp"].max()
    max_naive_error = conversion_summary["max_abs_naive_par_equals_zero_error_bp"].max()
    return f"""# PCA P&L diagnostics and ECB factor-attribution design

## Scope

This audit reads the existing PCA, swap-pricing and factor-regression files without rewriting them. All new outputs are isolated in `results/pca_diagnostics/`. The PCA sample remains 128 ECB events and four instruments. The factor section validates sources, coverage, event windows and curve-conversion mechanics but deliberately does **not** calculate factor-attributed swap P&L.

## PCA pricing-error audit

{chr(10).join(metric_lines)}

PCA-2 and PCA-3 materially reduce pricing error relative to a parallel-only shock. PCA-1 does not dominate parallel risk for every instrument, especially the 2Y payer and the DV01-neutral portfolio. Full annual-node KRD remains much closer to nonlinear repricing because PCA intentionally discards residual curve directions.

## Rolling variance explained

Every event uses 750 complete daily public-curve changes strictly before the event.

{chr(10).join(variance_lines)}

PC1 is the dominant level-like direction, PC2 is slope-like under the imposed sign convention, and PC3 is curvature-like. Their shares vary because each event has its own trailing estimation window.

## Paired bootstrap audit

The saved bootstrap resamples event dates and defines each contrast as squared error of method A minus squared error of method B. Positive intervals favor method B.

{chr(10).join(bootstrap_lines)}

{chr(10).join(instrument_bootstrap_lines)}

The PCA-2 versus PCA-3 improvement is therefore statistically clear for the 5Y payer and the pooled panel, but not separately for the 2Y payer, 10Y payer or DV01-neutral portfolio. Pooled results are not a substitute for the instrument-level rows.

## Zero-shock reconstruction

The saved PCA stage uses the standard affine reconstruction `mean + V_k V_k' (shock - mean)`. For an input shock of zero, the three-factor reconstruction has a mean RMS magnitude of **{zero3['affine_rms_bp_mean']:.4f} bp**, a 95th percentile of **{zero3['affine_rms_bp_p95']:.4f} bp**, and a maximum of **{zero3['affine_rms_bp_max']:.4f} bp** across the 128 rolling models. All ten components reconstruct zero to numerical precision (maximum RMS **{full['affine_rms_bp_max']:.2e} bp**).

This behavior is mathematically correct for an affine projection around the historical mean, but it does not preserve the economically natural origin when factors are truncated. An origin-preserving shock projection `V_k V_k' shock` maps zero to zero exactly. The original results are unchanged; an origin-preserving rerun should be treated as a separate robustness specification.

## Published factor coverage and windows

The common factor vintage is 2025-10-30. Target is a press-release factor normalized at 1M OIS. Timing, Forward Guidance and QE are press-conference factors normalized at 6M, 2Y and 10Y OIS. Positive values are restrictive surprises. Target, Timing and Forward Guidance are present for all 128 PCA events; QE is present for 99. The proposed complete four-factor sample therefore contains **{int(complete['available_among_128_pca_events'])} events**, from **{pd.Timestamp(complete['pca_event_start']).date()}** to **{pd.Timestamp(complete['pca_event_end']).date()}**. Missing QE observations are not zero and must not be imputed.

The workbook defines these windows:

- Through June 2022, release runs from the 13:25–13:35 median to 14:00–14:10; conference runs from 14:15–14:25 to 15:40–15:50; the monetary window runs from the first release baseline to the final conference endpoint.
- From July 2022, release runs from 13:55–14:05 to 14:30–14:40; conference starts at that same 14:30–14:40 median and ends at 15:55–16:05.

Accordingly, before July 2022 the component windows do not span the gap between their sampled endpoints: mean absolute `monetary - release - conference` is **{pre_gap['mean_abs_gap_bp']:.3f} bp** and its maximum is **{pre_gap['max_abs_gap_bp']:.3f} bp** on the eligible annual grid. From July 2022 the maximum discrepancy is **{post_gap['max_abs_gap_bp']:.2e} bp**, numerical zero. Pre-2022 factor contributions must therefore remain separate from an unattributed between-window residual.

## Par-to-zero conversion design

For each event and factor, the published factor value would multiply the appropriate annual regression-loading vector to form a fitted **par OIS** shock. These fitted par shocks must not be applied directly to zero rates. The validated conversion is:

1. Compute initial annual par rates from the event's proxy discount curve.
2. Add fitted par shocks in decimal units (`bp × 0.0001`).
3. Rebootstrap annual discount factors with the existing conventions.
4. Define zero shocks as shocked minus initial continuously compounded zero nodes.
5. For additive first-order factor contributions, apply an event-specific par-to-zero Jacobian to each factor vector. Compare their sum with a joint nonlinear rebootstrap and report the nonlinear remainder.

The reference-date unit-vector check reproduces every shocked par quote with maximum error **{max_par_error:.2e} bp**. Treating fitted par changes as zero changes would create errors as large as **{max_naive_error:.3f} bp** in this check.

## Identification and limitations

The starting curve is an ECB AAA sovereign proxy, not an OIS or €STR curve. The factor-loading regressions reuse rate maturities that helped construct and normalize the published factors. Resulting P&L contributions would be generated-regressor, descriptive decompositions, not independent causal estimates. A causal exercise requires outcomes not used in factor construction or a separate identification design. The factor-attribution calculation remains unexecuted pending an explicit decision on the 99-event complete sample, treatment of the pre-July-2022 window residual, and whether to use affine or origin-preserving PCA only as a separate robustness comparison.
"""


def run(
    pca_dir: Path,
    factor_dir: Path,
    swap_dir: Path,
    ois_path: Path,
    output_dir: Path,
) -> None:
    metrics = audit_error_metrics(pd.read_csv(pca_dir / "pca_pnl_error_metrics.csv"))
    variance = pd.read_csv(pca_dir / "pca_variance_explained.csv", parse_dates=["event_date"])
    variance_summary = summarize_explained_variance(variance)
    bootstrap = select_bootstrap_comparisons(
        pd.read_csv(pca_dir / "bootstrap_squared_error_differences.csv")
    )
    daily = pd.read_csv(
        pca_dir / "daily_aaa_zero_curve_changes.csv", parse_dates=["change_date"]
    ).set_index("change_date")
    eligibility = pd.read_csv(
        pca_dir / "pca_event_eligibility.csv", parse_dates=["event_date"]
    )
    zero_detail, zero_summary = zero_shock_reconstruction_test(daily, eligibility)

    merged = pd.read_csv(
        factor_dir / "merged_factor_ois_events.csv", parse_dates=["event_date"]
    )
    ois = pd.read_csv(ois_path, parse_dates=["event_date"])
    factor_events, factor_coverage, window_panel, gap_summary = audit_factor_coverage(
        eligibility, merged, ois
    )
    coefficients = pd.read_csv(factor_dir / "factor_coefficients_by_maturity.csv")
    zero_nodes = pd.read_csv(
        swap_dir / "event_zero_curve_shocks.csv", parse_dates=["event_date"]
    )
    conversion, conversion_summary, jacobian = par_to_zero_conversion_validation(
        coefficients, zero_nodes
    )
    design = factor_attribution_design()

    source_metadata = pd.read_csv(factor_dir / "factor_source_metadata.csv")
    local_factor_dir = Path("data/external/ea_mpd_factors")
    hashes = {
        path.name: _sha256(path)
        for path in local_factor_dir.glob("*2025-10-30.csv")
    }
    expected_hashes = {
        source_metadata["release_file"].iloc[0]: source_metadata["release_sha256"].iloc[0],
        source_metadata["conference_file"].iloc[0]: source_metadata["conference_sha256"].iloc[0],
    }
    if hashes != expected_hashes:
        raise ValueError("Local factor snapshot hashes do not match the audited vintage.")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "pca_error_metrics_by_instrument.csv", index=False)
    variance_summary.to_csv(output_dir / "pca_explained_variance_summary.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_selected_comparisons.csv", index=False)
    zero_detail.to_csv(
        output_dir / "zero_shock_reconstruction_by_event.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    zero_summary.to_csv(output_dir / "zero_shock_reconstruction_summary.csv", index=False)
    factor_events.to_csv(
        output_dir / "factor_event_coverage.csv", index=False, date_format="%Y-%m-%d"
    )
    factor_coverage.to_csv(
        output_dir / "factor_coverage_summary.csv", index=False, date_format="%Y-%m-%d"
    )
    window_panel.to_csv(
        output_dir / "factor_window_additivity_by_event_maturity.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    gap_summary.to_csv(output_dir / "factor_window_additivity_summary.csv", index=False)
    conversion.to_csv(
        output_dir / "factor_par_to_zero_conversion_validation.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    conversion_summary.to_csv(
        output_dir / "factor_par_to_zero_conversion_summary.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    jacobian.to_csv(
        output_dir / "reference_par_to_zero_jacobian.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    design.to_csv(output_dir / "factor_attribution_design.csv", index=False)

    _plot_metrics(metrics, output_dir / "pca_error_metrics.png")
    _plot_variance(variance, output_dir / "pca_explained_variance.png")
    _plot_bootstrap(bootstrap, output_dir / "bootstrap_selected_comparisons.png")
    _plot_zero_test(zero_detail, output_dir / "zero_shock_reconstruction.png")
    _plot_factor_conversion(conversion, output_dir / "factor_par_to_zero_conversion.png")
    (output_dir / "report.md").write_text(
        _report(
            metrics,
            variance_summary,
            bootstrap,
            zero_summary,
            factor_coverage,
            gap_summary,
            conversion_summary,
        ),
        encoding="utf-8",
    )
    metadata = {
        "pca_input_directory": str(pca_dir),
        "factor_input_directory": str(factor_dir),
        "swap_input_directory": str(swap_dir),
        "ois_input": str(ois_path),
        "pca_events": 128,
        "complete_four_factor_events": int(
            factor_events["proposed_complete_factor_sample"].sum()
        ),
        "factor_vintage": str(merged["factor_vintage"].dropna().iloc[0]),
        "factor_snapshot_sha256": hashes,
        "factor_pnl_attribution_executed": False,
        "original_results_modified": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-results", type=Path, default=Path("results/pca_pnl"))
    parser.add_argument(
        "--factor-results",
        type=Path,
        default=Path("results/policy_factor_regressions"),
    )
    parser.add_argument("--swap-results", type=Path, default=Path("results/swap_pricing"))
    parser.add_argument("--ois", type=Path, default=Path("results/ois_surprises.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/pca_diagnostics"))
    args = parser.parse_args()
    run(args.pca_results, args.factor_results, args.swap_results, args.ois, args.output)


if __name__ == "__main__":
    main()
