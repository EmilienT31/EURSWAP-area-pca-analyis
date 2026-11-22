# EA-MPD identified factor snapshot

This directory is an exact snapshot of the latest common press-release and
press-conference factor vintage observed in the user-supplied directory
`/Users/emilientran/Downloads/data` on 20 November 2026.

Selected vintage: **2025-10-30**

- `press_release_factors_2025-10-30.csv`
  - observed schema: `date,target`
  - SHA-256: `48c91fc9f52346bb564b49a29940b90c74b3b3a9edcce843a4aa269ae52bddd4`
- `press_conference_factors_2025-10-30.csv`
  - observed schema: `date,timing,fg,qe`
  - SHA-256: `930c74c30b5487b8d4a36da4016f1a2520e38a2514f22b28936fd6cae05b51ed`

The CSVs contain no embedded citation, unit, or normalization metadata. The
interpretation follows Altavilla, Brugnolini, Gürkaynak, Motto and Ragusa
(2019), *Measuring Euro Area Monetary Policy*, and its factor-rotation appendix:

- Target: press-release window; unit effect at 1M OIS.
- Timing: press-conference window; unit effect at 6M OIS.
- Forward Guidance: press-conference window; unit effect at 2Y OIS.
- QE: press-conference window; unit effect at 10Y OIS.

Positive factor values are restrictive surprises. Factor values are retained
exactly as supplied; no missing value is imputed. In particular, the QE series
is missing before its identified sample.
