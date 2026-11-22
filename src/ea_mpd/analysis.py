"""Transform and analyse validated EA-MPD OIS surprises."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA

from .data import EAMPDWorkbook


@dataclass(frozen=True)
class PCAResult:
    """Complete-case covariance PCA outputs."""

    explained_variance: pd.DataFrame
    loadings: pd.DataFrame
    scores: pd.DataFrame
    n_total_events: int
    n_complete_events: int


def extract_ois_surprises(workbook: EAMPDWorkbook) -> pd.DataFrame:
    """Return all event dates and OIS changes in tidy, maturity-ordered form."""

    parts: list[pd.DataFrame] = []
    for window, frame in workbook.sheets.items():
        schema = workbook.schemas[window]
        metadata = {
            column.source_name: column for column in schema.ois_columns
        }
        selected = [schema.date_column, *metadata]
        long = frame[selected].melt(
            id_vars=schema.date_column,
            var_name="source_column",
            value_name="surprise_bp",
        )
        long = long.rename(columns={schema.date_column: "event_date"})
        long.insert(1, "window", window)
        long["canonical_name"] = long["source_column"].map(
            lambda name: metadata[name].canonical_name
        )
        long["maturity"] = long["source_column"].map(
            lambda name: metadata[name].maturity_label
        )
        long["maturity_years"] = long["source_column"].map(
            lambda name: metadata[name].maturity_years
        )
        parts.append(long)

    result = pd.concat(parts, ignore_index=True)
    return result.sort_values(
        ["window", "event_date", "maturity_years"], ignore_index=True
    )


def descriptive_statistics(surprises: pd.DataFrame) -> pd.DataFrame:
    """Summarise signed and absolute surprises by window and maturity."""

    grouped = surprises.groupby(
        ["window", "canonical_name", "maturity", "maturity_years"],
        sort=False,
        dropna=False,
    )["surprise_bp"]
    stats = grouped.agg(
        total_events="size",
        n_observations="count",
        mean_bp="mean",
        median_bp="median",
        std_bp="std",
        min_bp="min",
        max_bp="max",
    ).reset_index()
    stats["missing_events"] = stats["total_events"] - stats["n_observations"]
    mean_abs = grouped.apply(lambda values: values.abs().mean()).rename("mean_abs_bp")
    quantiles = grouped.quantile([0.05, 0.95]).unstack().rename(
        columns={0.05: "p05_bp", 0.95: "p95_bp"}
    )
    stats = stats.merge(mean_abs.reset_index(), how="left").merge(
        quantiles.reset_index(), how="left"
    )
    order = [
        "window",
        "canonical_name",
        "maturity",
        "maturity_years",
        "total_events",
        "n_observations",
        "missing_events",
        "mean_bp",
        "median_bp",
        "std_bp",
        "mean_abs_bp",
        "min_bp",
        "p05_bp",
        "p95_bp",
        "max_bp",
    ]
    return stats[order].sort_values(
        ["window", "maturity_years"], ignore_index=True
    )


def run_pca(
    workbook: EAMPDWorkbook,
    event_sheet: str,
    selected_canonical_names: tuple[str, ...] | None = None,
) -> PCAResult:
    """Run centered, unstandardized PCA on complete event-window OIS curves.

    All maturities are in basis points, so covariance PCA preserves the relative
    economic volatility of short and long rates. Rows with any missing OIS value
    are excluded and their count is reported explicitly.
    """

    frame = workbook.sheets[event_sheet]
    schema = workbook.schemas[event_sheet]
    if selected_canonical_names is None:
        selected_columns = list(schema.ois_columns)
    else:
        if len(set(selected_canonical_names)) != len(selected_canonical_names):
            raise ValueError("Selected PCA maturities must be unique.")
        by_name = {column.canonical_name: column for column in schema.ois_columns}
        missing = [name for name in selected_canonical_names if name not in by_name]
        if missing:
            raise ValueError(f"PCA maturities are absent from the workbook: {missing}.")
        selected_columns = [by_name[name] for name in selected_canonical_names]

    source_columns = [column.source_name for column in selected_columns]
    complete = frame[[schema.date_column, *source_columns]].dropna()
    if len(complete) < 3:
        raise ValueError("PCA requires at least three complete event observations.")

    model = PCA()
    values = complete[source_columns].to_numpy()
    scores_values = model.fit_transform(values)
    component_names = [f"PC{i}" for i in range(1, model.n_components_ + 1)]

    # Component signs are mathematically arbitrary. Apply deterministic economic
    # conventions: PC1 has a positive average loading; later PCs have a positive
    # loading/maturity correlation when that correlation is defined.
    loadings_values = model.components_.T.copy()
    maturities = pd.Series(
        [column.maturity_years for column in selected_columns], dtype=float
    )
    for index in range(model.n_components_):
        if index == 0:
            flip = loadings_values[:, index].mean() < 0
        else:
            correlation = pd.Series(loadings_values[:, index]).corr(maturities)
            flip = pd.notna(correlation) and correlation < 0
        if flip:
            loadings_values[:, index] *= -1
            scores_values[:, index] *= -1

    explained = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_bp2": model.explained_variance_,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": model.explained_variance_ratio_.cumsum(),
        }
    )
    loadings = pd.DataFrame(
        loadings_values,
        columns=component_names,
    )
    loadings.insert(
        0, "maturity_years", [column.maturity_years for column in selected_columns]
    )
    loadings.insert(0, "maturity", [column.maturity_label for column in selected_columns])
    loadings.insert(
        0, "canonical_name", [column.canonical_name for column in selected_columns]
    )
    scores = pd.DataFrame(scores_values, columns=component_names)
    scores.insert(0, "event_date", complete[schema.date_column].to_numpy())
    return PCAResult(explained, loadings, scores, len(frame), len(complete))

