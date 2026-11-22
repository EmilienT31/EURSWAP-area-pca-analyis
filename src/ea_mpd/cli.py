"""Command-line entry point for the first EA-MPD analysis iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .analysis import descriptive_statistics, extract_ois_surprises, run_pca
from .data import find_monetary_event_sheet, load_ea_mpd


def _plot_mean_absolute_surprise(
    event_stats: pd.DataFrame, event_sheet: str, destination: Path
) -> None:
    ordered = event_stats.sort_values("maturity_years")
    positions = range(len(ordered))
    fig, axis = plt.subplots(figsize=(9, 5.25))
    axis.plot(
        positions,
        ordered["mean_abs_bp"],
        color="#125B8C",
        marker="o",
        linewidth=2,
    )
    axis.set_xticks(list(positions), ordered["maturity"])
    axis.set_xlabel("OIS maturity")
    axis.set_ylabel("Mean absolute surprise (basis points)")
    axis.set_title(f"Average OIS surprise magnitude — {event_sheet}")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_interpretation(
    destination: Path,
    source: Path,
    event_sheet: str,
    event_dates: pd.DataFrame,
    event_stats: pd.DataFrame,
    pca_result,
) -> None:
    event_only = event_dates.loc[event_dates["window"] == event_sheet]
    start = event_only["event_date"].min().date().isoformat()
    end = event_only["event_date"].max().date().isoformat()
    largest = event_stats.loc[event_stats["mean_abs_bp"].idxmax()]
    smallest = event_stats.loc[event_stats["mean_abs_bp"].idxmin()]
    pc1 = pca_result.explained_variance.iloc[0]["explained_variance_ratio"]
    first_three = pca_result.explained_variance.head(3).iloc[-1][
        "cumulative_explained_variance_ratio"
    ]
    complete_share = pca_result.n_complete_events / pca_result.n_total_events

    text = f"""# Economic interpretation of the generated outputs

Source workbook: `{source.name}`  
Full event window used for the maturity plot and PCA: `{event_sheet}`  
Event coverage: **{start} to {end}** ({len(event_only)} policy events)

## `event_dates.csv`

This is the event calendar extracted from every validated workbook window. Each row is an ECB policy-event date/window pair. It is the alignment key for any later market-data or P&L work; it is not itself a measure of policy stance.

## `ois_surprises.csv`

This is the tidy event-level dataset. `surprise_bp` is the signed change in the OIS rate over the workbook-defined intraday window, in basis points. A positive value means the market-implied euro short-rate path moved up (a hawkish surprise); a negative value means it moved down (a dovish surprise). Missing cells remain missing and are never replaced with zeros.

## `descriptive_statistics.csv`

The table separates signed location (`mean_bp`, `median_bp`) from shock size (`std_bp`, `mean_abs_bp`) and reports coverage explicitly. A near-zero signed mean can coexist with large shocks because hawkish and dovish surprises offset. In the full event window, **{largest['maturity']}** has the largest mean absolute move ({largest['mean_abs_bp']:.2f} bp), while **{smallest['maturity']}** has the smallest ({smallest['mean_abs_bp']:.2f} bp). This maturity profile shows where ECB announcements most strongly reprice the expected rate path.

## `average_ois_surprise_magnitude.png`

The figure plots `mean(abs(surprise_bp))`, not the absolute value of the mean. It therefore measures typical repricing intensity without cancellation between hawkish and dovish events. Short maturities mainly reflect revisions to the near-term expected policy rate; longer maturities increasingly include persistence, forward-guidance and term-premium responses.

## `pca_explained_variance.csv`

PCA compresses co-movement across the OIS curve. PC1 explains **{pc1:.1%}** of complete-case event-window variation; the first three PCs explain **{first_three:.1%}** cumulatively. A dominant PC1 is consistent with a broad level-like repricing of the expected policy path. Later components commonly capture rotations resembling slope and curvature, but their loadings—not their labels—should determine that interpretation.

## `pca_loadings.csv`

Loadings show which maturities move with each component. Similar same-sign PC1 loadings indicate a level factor. A component whose loadings change systematically from short to long maturities is slope-like; one that contrasts the belly with the ends is curvature-like. Signs are oriented deterministically for reproducibility but remain economically arbitrary.

## `pca_scores.csv`

Scores measure the exposure of each policy event to each empirical curve factor, in basis-point units. Large positive or negative scores flag events with strong factor-specific repricing; they are shocks, not returns or swap P&L.

## `pca_metadata.json`

PCA uses centered, unstandardized OIS changes because every maturity is measured in basis points. It uses {pca_result.n_complete_events} of {pca_result.n_total_events} events ({complete_share:.1%}) with observations at every maturity. Complete-case filtering prevents imputed values from becoming artificial policy surprises, but the PCA sample may tilt toward the later, more complete part of the database.
"""
    destination.write_text(text, encoding="utf-8")


def run(input_path: Path, output_dir: Path) -> None:
    workbook = load_ea_mpd(input_path)
    event_sheet = find_monetary_event_sheet(workbook)
    output_dir.mkdir(parents=True, exist_ok=True)

    surprises = extract_ois_surprises(workbook)
    event_dates = surprises[["event_date", "window"]].drop_duplicates().sort_values(
        ["event_date", "window"], ignore_index=True
    )
    stats = descriptive_statistics(surprises)
    pca_result = run_pca(workbook, event_sheet)

    event_dates.to_csv(output_dir / "event_dates.csv", index=False, date_format="%Y-%m-%d")
    surprises.to_csv(output_dir / "ois_surprises.csv", index=False, date_format="%Y-%m-%d")
    stats.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    pca_result.explained_variance.to_csv(
        output_dir / "pca_explained_variance.csv", index=False
    )
    pca_result.loadings.to_csv(output_dir / "pca_loadings.csv", index=False)
    pca_result.scores.to_csv(
        output_dir / "pca_scores.csv", index=False, date_format="%Y-%m-%d"
    )

    event_stats = stats.loc[stats["window"] == event_sheet].copy()
    _plot_mean_absolute_surprise(
        event_stats, event_sheet, output_dir / "average_ois_surprise_magnitude.png"
    )
    metadata = {
        "source_workbook": workbook.source.name,
        "event_sheet": event_sheet,
        "units": "basis points",
        "method": "centered covariance PCA (no variance standardization)",
        "missing_data": "complete cases across all inferred OIS maturities",
        "n_total_events": pca_result.n_total_events,
        "n_complete_events": pca_result.n_complete_events,
        "maturities": pca_result.loadings["maturity"].tolist(),
    }
    (output_dir / "pca_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    _write_interpretation(
        output_dir / "economic_interpretation.md",
        workbook.source,
        event_sheet,
        event_dates,
        event_stats,
        pca_result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate EA-MPD and analyse event-window OIS surprises."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("Dataset_EA-MPD.xlsx"), help="EA-MPD XLSX file"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"), help="output directory"
    )
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()

