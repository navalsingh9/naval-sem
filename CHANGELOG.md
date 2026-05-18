# NAVAL-SEM — Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [v0.4.0] — 2026-05-19 · Formative Measurement & Docs

### Added
- **Case studies library** (`cases/cases.json`) — structured repository of
  real-world SEM use cases across disciplines for in-app reference.
- **Cases documentation page** (`docs/cases/index.html`) — browsable HTML
  index of all bundled case studies.
- **SmartPLS comparison page** (`docs/compare/smartpls-vs-naval-sem.html`) —
  feature-by-feature comparison of naval-sem vs SmartPLS.
- **Build documentation** (`docs/building.md`) — instructions for building
  from source on Windows and Linux.
- **Sitemap** (`docs/sitemap.xml`) — full XML sitemap for docs discoverability.

### Changed
- **README** — restructured and updated to reflect v0.4 feature set.

---

## [v0.3.0] — 2025-05-11 · Measurement Completion

### Added
- **Average Variance Extracted (AVE)** — computed per latent variable using
  standardised loadings (`AVE = Σλ² / n`). Returned in `fit.ave`.
- **Composite Reliability (ρc)** — `(Σλ)² / ((Σλ)² + Σ(1 − λ²))` per LV.
  Returned in `fit.composite_reliability`.
- **Cronbach's Alpha** — standard covariance-matrix formula per LV,
  clamped to `[0, 1]`. Returned in `fit.cronbach_alpha`.
- **Fornell-Larcker Criterion** — full LV × LV matrix; diagonal = √AVE,
  off-diagonal = inter-construct correlation. Pass/fail verdict in
  `fit.fornell_larcker_pass`. Returned in `fit.fornell_larcker`.
- **Schema fields** — `ave`, `composite_reliability`, `cronbach_alpha`,
  `fornell_larcker`, `fornell_larcker_pass` added to `FitIndices`.
- **Validity tab** in the UI — colour-coded convergent validity table
  (AVE ✓/✗ >0.50, CR ✓/✗ >0.70, α ✓/✗ >0.70) and Fornell-Larcker matrix
  with overall discriminant validity verdict.
- **Download buttons** — "Download fit indices (.csv)", "Download validity
  (.csv)", and "Download full JSON" from the results panel.

### Fixed
- **SRMR always null** — semopy's `calc_stats` does not compute SRMR.
  Added manual computation via `Σ = ΛΦΛᵀ + Θ` reconstruction from
  parameter estimates, with residual variance fallback.
- **HTMT tab losing data on view switch** — HTMT result is now cached in
  `htmtData`; switching back to the HTMT tab re-renders from cache
  without a new API call.
- **HTMT not populated when running from Syntax tab** — `runFromSyntax()`
  was not calling `triggerHtmt()`. Fixed.
- **`FutureWarning` on single-element Series** — `_safe_float()` now
  calls `.iloc[0]` before `float()` when the value is a pandas Series.
- **`_compute_ave` NameError** — `def` line was accidentally dropped
  during a splice operation. Restored.
- **Unstandardised loadings in AVE / CR** — semopy ML fixes one loading
  per factor to 1.0 for identification, causing AVE and CR to exceed 1.
  `_extract_loadings()` now detects `|λ| > 1` and re-derives all loadings
  as `corr(indicator, construct_composite)` from the data.
- **`op == "=~"` filter finding no rows** — semopy CB-SEM writes `op = "~"`
  for all rows. Replaced op-based scan with a variable-name lookup driven
  by `parsed["measurement"]`.

---

## [v0.2.1] — 2025-05-09 · Stability & Security

### Changed
- Hardened API exception handling for CodeQL security compliance.
- Added pull request template for contributions.

### Added
- `CITATION.cff` for software citation.
- Methods template for SEM usage documentation.
- Validation and reproducibility documentation in `/docs`.

---

## [v0.2.0] — 2025-05-07 · Initial Release

### Added

**Backend (`app/`)**
- `engine.py` — core statistical engine wrapping semopy:
  - `fit_model()` — CB-SEM (ML), WLS, and PLS-SEM (falls back to CB-SEM
    if `semopy.PLS` unavailable). Returns full `ModelResult`.
  - `run_bootstrap()` — percentile bootstrap with configurable samples
    and seed. Reports BS mean, SE, 95% CI, and convergence rate.
  - `compute_htmt()` — Heterotrait-Monotrait ratio for all LV pairs,
    with pass/fail against the 0.90 threshold.
  - `export_as_code()` — exports model as runnable R/lavaan, Python/semopy,
    or `.lav` syntax file.
- `parser.py` — lavaan-syntax parser supporting `=~`, `~`, and `~~`
  operators. Extracts latent variables, observed indicators, structural
  paths, and covariances. Includes Excel (`.xlsx`) and SPSS (`.sav`)
  file parsers.
- `schemas.py` — Pydantic response models:
  - `PathParameter` — per-path estimate, SE, z, p, 95% CI, significance.
  - `FitIndices` — CFI, RMSEA, SRMR, χ², df, p, AIC, BIC, R²,
    plus threshold verdict flags (`cfi_good`, `rmsea_acceptable`, etc.).
  - `BootstrapResult`, `HTMTEntry`, `HTMTResult`, `ModelResult`.
- `main.py` — FastAPI application:
  - `GET  /health` — server liveness check.
  - `POST /upload/preview` — column preview for CSV / Excel / SPSS.
  - `POST /run` — fit model, optionally run bootstrap inline.
  - `POST /bootstrap` — standalone bootstrap endpoint.
  - `POST /htmt` — standalone HTMT endpoint.
  - `POST /validate-syntax` — lavaan syntax validation without fitting.
  - `POST /export` — code export returning a plain-text file download.

**Frontend (`static/index.html`)**
- Visual model builder — drag-and-drop canvas for placing latent
  variables, observed indicators, and drawing measurement / structural
  paths. Auto-generates lavaan syntax from the canvas.
- Syntax editor — editable lavaan textarea synced with the canvas;
  supports running analysis directly from syntax.
- HTMT view — separate tab showing HTMT matrix with pass/fail per pair.
- Results panel with three tabs:
  - **Fit indices** — colour-coded chips (green/amber/red) for CFI,
    RMSEA, SRMR, χ², AIC, BIC, R².
  - **Parameters** — full table with estimate, SE, z, p-value, 95% CI,
    significance flag.
  - **Path chart** — horizontal bar chart of structural path estimates.
- Bootstrap results tab (shown when bootstrap N > 0).
- File upload supporting CSV, Excel, and SPSS with column-pill preview.
- Algorithm selector (PLS-SEM / CB-SEM / WLS).
- Missing data handler (listwise deletion / mean imputation).
- Export code dropdown (R/lavaan, Python/semopy, JASP/.lav) —
  copies generated code to clipboard.
- Server health indicator with 5-second polling.

**Packaging**
- Windows installer via PyInstaller + NSIS.
- `launcher.py` for desktop (pywebview) and browser modes.

---

*Roadmap: v0.4 (VIF, f², indirect effects) → v1.0 (fsQCA, APA reports).
See [NAVAL-SEM Release Plan](docs/) for full dependency-ordered feature list.*
