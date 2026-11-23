# From ECB Monetary Policy Surprises to EUR Rates Derivatives P&L

This first iteration validates the ECB's EA-MPD workbook, extracts event dates
and intraday OIS rate changes, produces descriptive statistics and a maturity
profile of mean absolute surprises, and runs PCA on the full monetary-event
window. It deliberately does **not** price swaps or interpret rate changes as
P&L.

The implementation was written against the observed workbook structure dated
4 November 2025: one `Notes` sheet and three 315-event data sheets. Rather than
hard-coding the complete header list, the loader discovers the date column,
parses maturity-labelled OIS columns, and uses definitions in `Notes` to resolve
the observed `OIS_SW`/`OIS1W` inconsistency. Ambiguity is an error.

## Reproduce

Requirements: `uv` and Python 3.12.

1. Put `Dataset_EA-MPD.xlsx` at the project root, or pass another path with
   `--input`.
2. Install the locked environment and run the analysis:

   ```bash
   uv sync --python 3.12
   uv run ea-mpd-analyze --input Dataset_EA-MPD.xlsx --output results
   ```

3. Run the tests:

   ```bash
   uv run pytest
   ```

`results/economic_interpretation.md` explains the economics and limitations of
every generated output. Missing OIS observations are retained in the extracted
CSV and excluded—not imputed—for PCA.

## Robustness analysis

The separate robustness command preserves the original outputs and writes its
common-sample comparison and alternative maturity-grid PCA under
`results/robustness/`:

```bash
uv run ea-mpd-robustness \
  --input Dataset_EA-MPD.xlsx \
  --output results/robustness
```

It compares the original maturity-specific samples with a common complete
16-maturity sample, then compares the 16-maturity PCA with a longer-sample PCA
on 1M, 3M, 6M, 1Y, 2Y and 3Y. Both use centered, unstandardized basis-point
changes and no missing-value imputation.

## Identified policy factors

The distributed EA-MPD workbook does not contain the paper's rotated factor
scores. The factor regression workflow uses a documented common vintage of the
separate press-release and press-conference CSVs:

```bash
uv run ea-mpd-factor-regressions \
  --input Dataset_EA-MPD.xlsx \
  --factor-dir data/external/ea_mpd_factors \
  --vintage 2025-10-30 \
  --output results/policy_factor_regressions
```

Target is matched only to press-release OIS changes. Timing, Forward Guidance,
and QE enter jointly against press-conference OIS changes. Missing factor and
OIS observations remain visible in the exported merged panel; the models use
window-specific common samples across all maturities with HC3 standard errors.
The loader validates the observed schemas and never relabels PCA scores.

## Project layout

```text
src/ea_mpd/data.py      workbook loading, schema inference, validation
src/ea_mpd/analysis.py  tidy extraction, statistics, covariance PCA
src/ea_mpd/cli.py       reproducible output pipeline and figure
tests/                  focused unit and integration tests
results/                generated outputs (not version-controlled)
```

## Stylised EUR OIS pricing and curve risk

The pricing stage is separate from the econometric modules. It downloads only
the ECB AAA (`G_N_A`) curve nodes required through 10Y, validates the SDMX
schema, and stores a compact normalized cache rather than the large ECB bulk
file:

```bash
uv run ea-mpd-download-aaa-curve \
  --start 2004-09-06 \
  --end 2025-10-30 \
  --output data/external/ecb_aaa_curve
```

Run the pricing and real-event counterfactual repricing with:

```bash
uv run ea-mpd-swap-pricing \
  --curve data/external/ecb_aaa_curve/aaa_spot_rates_3m_10y.csv \
  --surprises results/ois_surprises.csv \
  --output results/swap_pricing
```

The core engine uses NumPy/SciPy and exposes continuously compounded zero
rates, log-discount-factor interpolation, forwards, par rates, bootstrapping,
parallel DV01 and triangular key-rate risk. QuantLib is an optional independent
validation dependency:

```bash
uv sync --group validation
uv run --group validation pytest
```

The ECB curve is a AAA sovereign proxy, not an €STR OIS curve. EA-MPD OIS
changes are applied to implied par swap nodes before re-bootstrapping; they are
not applied directly to zero rates. Full conventions and limitations are
recorded in `results/swap_pricing/methodology.md`.

### Event-level risk decompositions

The extension compares matched par-annuity, uniform zero-curve and annual-node
key-rate approximations across every event with an exact same-date public curve
and complete 1Y--10Y EA-MPD shocks:

```bash
uv run ea-mpd-swap-risk-experiments \
  --curve data/external/ecb_aaa_curve/aaa_spot_rates_3m_10y.csv \
  --surprises results/ois_surprises.csv \
  --output results/swap_pricing
```

It also constructs a 2Y payer/10Y receiver portfolio whose 10Y notional is
chosen on each event date to cancel parallel zero-rate DV01. Excluded dates and
missing observations remain visible in an eligibility audit; curves are never
carried backward or forward and surprises are never imputed.

### PCA P&L attribution

The rolling PCA extension preserves every earlier output and writes only to
`results/pca_pnl/`:

```bash
uv run ea-mpd-pca-pnl \
  --curve data/external/ecb_aaa_curve/aaa_spot_rates_3m_10y.csv \
  --pricing-results results/swap_pricing \
  --output results/pca_pnl
```

For every eligible event it estimates centered, unstandardized covariance PCA
on up to 750 complete public-curve daily changes strictly before the event
(minimum 250), projects the observed annual zero shock onto one, two and three
components, and applies the existing event-specific zero-node KRD vector. The
stage compares PCA with parallel-only, full-KRD and nonlinear repricing on one
common event/instrument sample, including the DV01-neutral curve portfolio.

Audit the saved PCA results and prepare the window-consistent ECB factor P&L
design without executing that attribution:

```bash
uv run ea-mpd-pca-diagnostics --output results/pca_diagnostics
```

This writes a separate diagnostics report, instrument-level error tables,
rolling explained-variance summaries, selected paired-bootstrap comparisons,
a zero-input PCA reconstruction check, published-factor coverage, event-window
additivity checks and a validated par-to-zero conversion example. Existing PCA,
pricing and factor-regression outputs are read-only inputs.

