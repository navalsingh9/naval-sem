# NAVAL-SEM — Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [v0.5.1] — 2026-05-26 · Security & Code Quality

### Fixed
- **Bandit B110 (try/except/pass)** — all bare `except Exception:` blocks that
  silently swallowed errors with `pass` now capture the exception as `_e` and
  emit `logger.debug(...)` before passing. Affects `app/engine.py` (5 locations).
- **Bandit B112 (try/except/continue)** — all bare `except Exception:` blocks
  followed by `continue` now capture the exception as `_e`. Affects
  `app/engine.py` (11 locations) and `app/pls.py` (4 locations).
- **`app/pls.py` missing logger** — added `import logging` and
  `logger = logging.getLogger("naval_sem.pls")` to support the above fixes.
- Resolves all 14 open Code Scanning alerts on GitHub (Bandit, severity: Note).

---

## [v0.5.0] — 2026-05-25 · Predictive Relevance + CMB

### Added

**Predictive relevance suite — `POST /predict`**
- **Q² Blindfolding** — Stone-Geisser Q² via omission loop (default D=7).
  `Q² = 1 − SSE/SSO` per endogenous LV. Benchmarks: none <0.02, small ≥0.02,
  medium ≥0.15, large ≥0.35. Returned in `PredictResult.q2`.
- **PLSpredict** — k-fold cross-validation (default k=10) comparing model RMSE/MAE
  against a naive LM baseline per indicator of each endogenous LV.
  `Q²_predict = 1 − (RMSE_model / RMSE_lm)²`. Returned in `PredictResult.plspredict`.
- **CVPAT** (Cross-Validated Predictive Ability Test, Liengaard et al. 2021) —
  one-sample t-test on per-observation loss difference (LM − model).
  Significant only when `cvpat_statistic > 1e-6` to guard against floating-point
  near-zero. Returned in `PredictResult.cvpat`.
- **CMB Marker Variable Analysis** — `POST /cmb` — Lindell & Whitney (2001)
  method: correlates a theoretically unrelated marker variable with all substantive
  indicators. Flags `cmb_concern = True` when max |r| > 0.20.
  Returned as `CMBMarkerResult`.

**Schema additions (`schemas.py`)**
- `Q2Entry`, `PLSPredictEntry`, `CVPATResult`, `CMBMarkerResult`, `PredictResult`

**Predictive tab in UI**
- New **Predictive** tab (between Effects and Parameters) auto-populates after
  every model run — no extra button needed.
- Q² table with colour-coded relevance badges (large/medium/small/none).
- PLSpredict table with RMSE model vs LM baseline per indicator, ✓/✗ verdict.
- CVPAT table with statistic, p-value, and verdict per endogenous LV.
- CMB panel — enter a marker column name in the sidebar and press Enter; result
  appears in the Predictive tab without re-running the model.
- All data downloadable via **Downloads → v0.5 Predictive Relevance (.csv)** and
  **CMB analysis (.csv)** (CMB button appears only after a CMB run).

### Fixed
- **CVPAT `significant: true` when statistic ≈ 0** — with large fold-level
  sample sizes, a statistically significant t-test was firing on effectively zero
  loss differences. Added `mean_diff > 1e-6` guard.

---

## [v0.4.2] — 2026-05-22 · Distribution Pipeline Cleanup

### Changed
- **Removed SourceForge release pipeline** — `.github/workflows/release.yml`
  no longer pushes builds to SourceForge automatically. Distribution is now
  handled via the SourceForge portal's GitHub connector, keeping the workflow
  file lean and removing the dependency on SF credentials in CI secrets.
- Cleaned up `.gitignore` entries related to the removed pipeline artifacts.

---

## [v0.4.1] — 2026-05-22 · Effects, Significance & UI Polish

### Added
- **VIF (Variance Inflation Factor)** per indicator per LV block —
  `VIF_i = 1 / (1 − R²_i)` from OLS regression of each indicator on all others
  in its block. Thresholds: <3.3 strict, <5.0 acceptable. Auto-computed in
  `fit_model()`. Returned in `ModelResult.vif`.
- **Cohen's f² effect size** per structural path —
  `f² = (R²_full − R²_reduced) / (1 − R²_full)` using OLS composite scores
  (no semopy refitting, works for CB-SEM/PLS/WLS). Benchmarks: negligible <0.02,
  small ≥0.02, medium ≥0.15, large ≥0.35. Returned in `ModelResult.f2`.
- **Outer weight significance** — bootstrap significance test for all measurement
  loadings/weights. Reports BS mean, SE, 95% percentile CI, t-stat = estimate/BS_SE.
  Only runs when `bootstrap_n > 0`. Returned in `ModelResult.outer_weights`.
- **Indirect effects decomposition** — `POST /indirect` — DFS path tracing for
  all variable pairs with paths ≥ 2 edges (mediation). Bootstrapped 95% CIs.
  Total effects matrix (direct + indirect, structural vars only).
  Returned as `IndirectResult`.
- **Effects tab in UI** — VIF, f², outer weight significance, indirect effects,
  and total effects matrix all in one tab with colour-coded verdicts.
- **PLS significance back-fill from bootstrap CIs** — PLS-SEM produces no
  analytical p-values; when bootstrap is run, CIs replace the p-value sentinel
  (`0.001` = significant, `0.999` = not) for all structural paths. Triggers
  whenever any structural path has `p ≥ 0.999`, regardless of `use_pls` flag.
- **Downloads tab** — single consolidated tab replacing scattered per-tab download
  buttons. Sections: Fit & Model, Measurement Validity, v0.4 Effects, Bootstrap,
  Full Export (R/Python/JASP), v0.5 Predictive Relevance.

### Fixed
- **`_extract_loadings` op-agnostic rewrite** — semopy CB-SEM writes `op = "~"`
  for all rows; replaced op-based scan with variable-name lookup against
  `parsed["measurement"]`, with bidirectional `(left, right) → estimate` index.
- **Unstandardised loadings causing AVE/CR > 1.0** — detected `|λ| > 1` and
  switched to `corr(indicator, construct_composite)` from the data.
- **Path chart showing measurement rows** — chart now filters to LV→LV structural
  paths only using `latent_variables` set.
- **HTMT tab losing data on view switch** — cached in `htmtData`; re-renders on
  tab switch without a new API call.
- **`FutureWarning` on single-element Series** — `_safe_float()` calls `.iloc[0]`
  before `float()` when value is a pandas Series.
- **`_compute_ave` NameError** — `def` line dropped during splice; restored.
- **CVPAT `significant: true` on zero statistic** — added `mean_diff > 1e-6` guard.

---

## [v0.4.0] — 2026-05-18 · Formative + Effect Sizes (initial release)

### Added
- Schema: `VIFEntry`, `F2Entry`, `IndirectEffect`, `IndirectResult`,
  `OuterWeightEntry` added to `schemas.py`.
- `ModelResult` extended with `vif`, `f2`, `indirect`, `outer_weights` fields.
- `main.py` version bumped to `0.4.0`.
- `POST /indirect` endpoint.

---

## [v0.3.1] — 2026-05-11 · Bug fixes

### Fixed
- **SRMR always null** — semopy's `calc_stats` omits SRMR; added manual
  computation via `Σ = ΛΦΛᵀ + Θ` reconstruction, with residual variance fallback.
- **`runFromSyntax` not calling `triggerHtmt()`** — fixed; both canvas-run and
  syntax-run now trigger HTMT.
- **Canvas edge labels not showing for some LVs** — replaced `parameters`-based
  matching (fragile due to semopy op column) with `outer_weights` lookup for
  measurement edges and `parameters` for structural edges only.
- **Syntax tab blank after model run** — restored Syntax button, `syntax-view` div,
  and `switchView` handler for the syntax case.
- JS syntax errors from dropped `if(kind===...)` guards in `downloadResults` — fixed.

---

## [v0.3.0] — 2026-05-11 · Measurement Completion

### Added
- **Average Variance Extracted (AVE)** — `AVE = Σλ² / n` per LV. In `fit.ave`.
- **Composite Reliability (ρc)** — `(Σλ)² / ((Σλ)² + Σ(1−λ²))`. In `fit.composite_reliability`.
- **Cronbach's Alpha** — covariance-matrix formula, clamped `[0, 1]`. In `fit.cronbach_alpha`.
- **Fornell-Larcker Criterion** — diagonal = √AVE, off-diagonal = inter-construct r,
  with overall pass/fail verdict. In `fit.fornell_larcker` / `fit.fornell_larcker_pass`.
- **Schema fields** — `ave`, `composite_reliability`, `cronbach_alpha`,
  `fornell_larcker`, `fornell_larcker_pass` on `FitIndices`.
- **Validity tab in UI** — colour-coded convergent validity table (✓/✗ vs thresholds)
  and Fornell-Larcker matrix with discriminant validity verdict badge.
- **Second counter animation** in spinner — elapsed seconds + rotating status messages
  ("Fitting model…", "Running bootstrap samples…", etc.).
- **Drag-to-resize results panel** — pill handle at top of panel; drag up/down.
- **Tab active state** — fixed using `data-tab` attribute matching.

### Fixed
- **SRMR null** — manual SRMR computation added as fallback.
- **Unstandardised loadings** — `_extract_loadings` auto-detects `|λ| > 1` and
  recomputes from composite correlations.
- **`op == "=~"` filter finding no rows** — replaced with variable-name lookup.

---

## [v0.2.1] — 2026-05-07 · Stability & Security

### Changed
- Hardened API exception handling for CodeQL security compliance.
- Added pull request template for contributions.
- `CITATION.cff` for software citation.
- Methods template and reproducibility documentation in `/docs`.

---

## [v0.2.0] — 2026-05-07 · Initial Release

### Added

**Backend (`app/`)**
- `engine.py` — CB-SEM (ML), WLS, PLS-SEM (falls back to CB-SEM), bootstrapping,
  HTMT, and code export.
- `parser.py` — lavaan-syntax parser (`=~`, `~`, `~~`); Excel and SPSS file parsers;
  robust CSV/TSV parser with delimiter sniffing.
- `schemas.py` — `PathParameter`, `FitIndices` (CFI, RMSEA, SRMR, χ², AIC, BIC,
  R², verdict flags), `BootstrapResult`, `HTMTEntry`, `HTMTResult`, `ModelResult`.
- `main.py` — `GET /health`, `POST /upload/preview`, `POST /run`, `POST /bootstrap`,
  `POST /htmt`, `POST /validate-syntax`, `POST /export`.

**Frontend (`static/index.html`)**
- Visual model builder (drag-and-drop canvas, auto-generates lavaan syntax).
- Syntax editor with live canvas sync and Run from Syntax.
- HTMT view.
- Results panel: Fit indices, Validity, Effects, Predictive, Parameters, Path chart,
  Bootstrap, Downloads tabs.
- File upload (CSV / Excel / SPSS) with column-pill preview.
- Algorithm selector (PLS-SEM / CB-SEM / WLS), bootstrap N, missing data handler.
- Export code (R/lavaan, Python/semopy, JASP/.lav).
- Server health indicator.
- Undo/Redo/PNG download/Clear canvas toolbar on canvas.
- Selected-node floating badge with inline Delete.

**Packaging**
- Windows installer via PyInstaller + NSIS.
- `launcher.py` for desktop (pywebview) and browser modes.

---

*Roadmap: v0.6 (HOC, MICOM, MGA) → v1.0 (fsQCA, APA reports).
See [NAVAL-SEM Release Plan](docs/) for full dependency-ordered feature list.*
