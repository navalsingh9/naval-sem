# NAVAL-SEM — Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

**Formative construct (Mode B) support — PLS-SEM**
- Added `<~` lavaan-style operator for formative measurement blocks (e.g. `Quality <~ q1 + q2 + q3`), alongside the existing `=~` reflective operator. Checked before `=~` in `parse_lavaan()`'s operator chain so `<~` lines aren't swallowed by the plain `~` branch; `preprocess_lavaan()` now also treats a trailing `<~` as a line-continuation marker for multi-line formative blocks.
- `parse_lavaan()` returns two new keys: `formative_lvs` (LVs declared with `<~`) and `construct_modes` (`{lv: "A"|"B"}` for every LV in the model, defaulting to `"A"` when a construct has no formative declaration).
- `PLSEstimator`'s outer-weight update now branches per construct on `construct_modes`: Mode B blocks solve `w = (X'X + ridge·I)⁻¹ X'η_inner` (ridge = 1e-4, falling back to the unregularised solve only if that raises `LinAlgError`), normalised so `Var(η) = 1`; Mode A blocks are unchanged. `PLSResult` gains a `construct_modes` field carrying the per-LV mode through to the response.
- Affects `app/parser.py`, `app/pls.py`.

### Changed

- **`compute_vif()` now skips Mode A (reflective) constructs** — VIF/multicollinearity is only a meaningful diagnostic for formative blocks; reflective indicators are expected to correlate highly since they share a common cause. Models composed entirely of reflective constructs now return an empty VIF list instead of one entry per indicator. Affects `app/engine.py`.

### Fixed

- **Gaussian Copula bootstrap CI always `None` / `copula_significant` always `False`** — in `compute_gaussian_copula()`, the bootstrap loop computed the resampled copula coefficient (`c_bs`) on every iteration but never appended it to `bs_cop_coef`, so `_ci_from_bootstrap()` always received an empty list. Added the missing `bs_cop_coef.append(float(c_bs[-1]))`. Affects `app/engine.py`.
- **Fornell-Larcker off-diagonal used mean indicator correlation instead of LV composite correlation** — `_compute_fornell_larcker()` computed each off-diagonal cell as the mean of all cross-indicator correlations between two LV blocks rather than the Pearson r between LV composite scores required by Fornell & Larcker (1981). Added an optional `composites` parameter and a `_phi()` helper that correlates composite scores directly when supplied, falling back to the previous mean-cross-indicator behaviour otherwise; `_compute_measurement_validity()` now builds composites via `_build_composites()` and passes them through. Affects `app/engine.py`.
- **Both-paths (Hayes Model 58/59) combined indirect effect never computed** — when a model had both an a-path interaction (`X*W`) and a b-path interaction (`M*W`) for the same `X→M→Y` chain, each was returned as a separate `ModMediationPath` entry and the combined conditional indirect effect was never computed. `run_mod_mediation()` now detects such chains and appends a `moderated_path="both"` entry using `IE(w) = (a + a₃w)(b + b₃w)` and `imm = a₃b + ab₃`; CIs on this combined entry are `None` pending a simultaneous a/b bootstrap. Affects `app/engine_mod_mediation.py`.

---

## [v1.0.0] — 2026-06-27 · fsQCA + Reporting (Release Gate)

### Added

**fsQCA — fuzzy-set Qualitative Comparative Analysis**
- Added `app/fsqca.py` — calibration, necessity analysis, truth table construction, and Quine-McCluskey Boolean minimisation.
- Complex, parsimonious, and intermediate solution types, each with raw/unique coverage and consistency per term.
- Added `POST /fsqca` endpoint.
- New schemas: `NecessityEntry`, `TruthTableRow`, `FsQCAConfigTerm`, `FsQCASolution`, `BubbleChartPoint`, `FsQCAResult`.
- Bubble-chart coincidence output for frontend visualisation of fuzzy-set membership.

**APA 7th Edition Reporting**
- Added `app/report.py` — aggregates results from all prior analysis engines into submission-ready tables.
- Word (`.docx`) export of measurement model, discriminant validity, structural model, and indirect effects tables formatted to journal submission standards.
- Added `POST /report` endpoint.

### Changed

- **Schema freeze** — `schemas.py` public result models are now marked stable for semantic versioning. No fields may be renamed, removed, or narrowed in patch/minor releases; new optional fields only. Added `model_config = ConfigDict(extra="ignore")` to all frozen result models (`ModelResult`, `BootstrapResult`, `ModerationResult`, `IPMAResult`, `NCAResult`, `NCAESSEResult`, `FIMIXResult`, `PLSPOSResult`, `MGAResult`, `ModMediationResult`, `FsQCAResult`, `ScaleDevelopmentResult`, `CVIResult`).
- Version bumped to `1.0.0`.
- Updated `.github/workflows/release.yml` release notes for the v1.0 milestone.

---

## [v0.9.0] — 2026-06-22 · Validity Extension: CVI, EFA, Nomological, Invariance & NCA-ESSE

### Added

**Content Validity Index (CVI)**
- Added `POST /cvi` endpoint — item-level (I-CVI) and scale-level (S-CVI/Ave, S-CVI/UA) content validity indices.
- Expert ratings accepted as input matrix; Polit & Beck (2006) thresholds applied automatically.
- Returns per-item verdict and overall scale-level CVI with pass/fail classification.

**Exploratory Factor Analysis (EFA)**
- Added `POST /efa` endpoint — principal-axis factoring with oblique (promax) and orthogonal (varimax) rotation.
- Kaiser criterion and scree-plot eigenvalues returned for factor retention guidance.
- Factor loadings, communalities, and percentage of variance explained per factor.
- Supports pre-specification of number of factors or automatic extraction.

**Nomological Validity**
- Added `POST /nomological` endpoint — bivariate correlation matrix across theoretical constructs.
- Directional hypotheses checked against sign expectations supplied in the request body.
- Returns hypothesis verdict (supported / not supported) alongside correlation coefficients and p-values.

**Measurement Invariance (MICOM — extended)**
- Added `POST /invariance` endpoint — full MICOM workflow for partial and full measurement invariance across groups.
- Step 1: Configural invariance check. Step 2: Compositional invariance (permutation test). Step 3: Equality of mean composites and variances.
- Returns permutation p-values, confidence intervals, and an overall invariance verdict.

**NCA Effect Size Sensitivity Extension (NCA-ESSE)**
- Added `app/nca_esse.py` — `compute_nca_esse()` implements threshold-removal sensitivity sweep over CE-FDH and CR-FDH ceiling techniques.
- Joint-uniform benchmark and permutation significance test included.
- Added `POST /nca-esse` endpoint.
- New schemas: `NCAESSEThresholdPoint`, `NCAESSEEntry`, `NCAESSEResult`.
- Implements Becker, Richter, Ringle & Sarstedt (2026). J. Bus. Res. 206, 115920.

### Fixed

- `nca.py`: corrected `_ce_fdh` return-type annotation (documented as 3-tuple, actually returns 4-tuple including `ceil_pts`) and updated its docstring. No logic change — the function already returned the correct value.
- `schemas.py` / `engine.py`: renamed `NomologicalResult`'s `construct` field to `construct_name` (alias=`'construct'`) to silence the pydantic `BaseModel.construct()` shadow warning at startup. JSON wire format is unchanged — FastAPI serialises by alias.

### Changed

- Version bumped to `0.9.0`.
- Updated `.github/workflows/release.yml` for v0.9 release pipeline.
- Updated `naval_sem.spec` for PyInstaller packaging with new modules.


---

## [v0.8.0] — 2026-06-14 · Advanced Robustness: FIMIX, PLS-POS, PDF Export & Versioning

### Added

**FIMIX-PLS Segmentation**
- Added `app/fimix.py` — EM-based finite mixture segmentation over K latent segments.
- Added AIC, BIC and CAIC model-selection criteria for optimal segment count.
- Added segment membership assignment and per-segment path coefficient reporting.
- Targets unobserved heterogeneity detection as required for robustness assessment.

**PLS-POS Segmentation**
- Added `app/plspos.py` — prediction-oriented segmentation building on FIMIX infrastructure.
- Added response-based segmentation for prediction-focused structural models.

**PDF Export**
- Added `app/export_pdf.py` — full results export to PDF using bundled DejaVu Sans font family.
- Added `fonts/` directory containing DejaVu Sans (regular, bold, oblique, condensed variants).
- Enables submission-ready report generation directly from the application.

**Versioning**
- Added `app/version.py` — centralised version string for consistent runtime reporting.

**Backend**
- Updated `app/engine.py`, `app/engine_mga.py`, `app/engine_moderation.py`, `app/engine_mod_mediation.py`, `app/engine_ipma.py`, `app/engine_utils.py` with robustness-related enhancements.
- Extended `app/schemas.py` with robustness checks schema block.
- Updated `app/main.py` with new endpoints and version integration.
- Updated `app/parser.py` and `app/pls.py` for robustness workflow support.
- Updated `app/nca.py` with additional robustness-related analysis support.

### Changed

- Updated `static/index.html` to surface FIMIX, PLS-POS and PDF export in the UI.
- Updated `pyproject.toml` and `requirements.in` to reflect new dependencies.
- Updated `.github/workflows/release.yml` for v0.8 release pipeline.
- Updated `naval_sem.spec` for PyInstaller packaging with fonts and new modules.
- Version bumped to `0.8.0`.


---

## [v0.7.0] � 2026-06-08 � Moderation, IPMA, NCA & Conditional Process Analysis

### Added

**Moderation Analysis**
- Added dedicated moderation engine with product-of-composites estimation.
- Added automatic interaction-term detection using lavaan-style X*M syntax.
- Added bootstrap confidence intervals for interaction effects.
- Added simple-slope analysis at low (-1 SD), mean and high (+1 SD) moderator values.
- Added ?R� and Cohen's f� effect-size reporting for moderation effects.
- Added manifest-variable OLS fallback for moderation models without latent constructs.

**Importance�Performance Map Analysis (IPMA)**
- Added IPMA engine for construct prioritisation.
- Added total-effect importance computation.
- Added 0�100 performance rescaling for latent variable scores.
- Added target-construct analysis endpoint and reporting.

**Necessary Condition Analysis (NCA)**
- Added CE-FDH and CR-FDH ceiling techniques.
- Added permutation-based significance testing.
- Added effect-size classification and bottleneck analysis support.
- Added NCA result schemas and reporting structures.

**Moderated Mediation / Conditional Process Analysis**
- Added conditional indirect-effect estimation.
- Added Index of Moderated Mediation (IMM).
- Added support for Hayes PROCESS-style Models 7, 14 and 58/59.
- Added bootstrap confidence intervals for conditional indirect effects.

**Backend**
- New pp/engine_moderation.py.
- New pp/engine_mod_mediation.py.
- New pp/engine_ipma.py.
- New pp/engine_utils.py.
- Extended API routes and schema definitions for all new analytical workflows.

### Improved

- Added Model Summary reporting object for simplified interpretation of results.
- Added reproducibility fingerprint generation using SHA-256 hashes.
- Added automatic reverse scoring via 
Variable naming convention.
- Improved parser handling for advanced moderation and conditional-process syntax.
- Improved SEM reporting and result aggregation.
- Expanded API documentation and endpoint coverage.

### Changed

- Refactored v0.7 functionality into dedicated analysis engines.
- Removed legacy engine_v07.py.
- Updated frontend (static/index.html) to support new workflows and reporting views.
- Version bumped to  .7.0.
---

## [0.6.1] — 2026-06-03
 
### Fixed
 
- **Syntax → Builder tab now renders the diagram.** Typing lavaan syntax in the Syntax tab and clicking Builder produced a blank canvas. Root cause: `generateCanvasFromSyntax()` existed but was never called on tab switch. `showCanvasMode('builder')` now reads the syntax textarea and invokes it before triggering `resize()` → `draw()`.
- **Structural paths (`~`) now draw regardless of line order.** `generateCanvasFromSyntax` used a single pass, so `~` edges were silently dropped when they appeared before the `=~` lines that create their nodes. Rewrote to two passes: Pass 1 creates all LV + indicator nodes from `=~` lines; Pass 2 processes `~` lines with all nodes guaranteed to exist. Any label referenced in `~` but absent from `=~` is auto-created as an LV node.
### Improved
 
- Inline comments (`# ...`) are now stripped from syntax before parsing, so annotated syntax files parse cleanly.
- Free-parameter prefixes (e.g. `0.5*x1`, `1*x2`) are stripped from indicator and predictor labels during canvas generation.
- Removed dead code from `showCanvasMode` syntax branch (`src` alias pointing to same element as `ta`, unused `canvasSrc` querySelector).


---

## [v0.6.0] — 2026-06-02 · Higher-Order Constructs, MICOM & MGA

### Added

**Higher-Order Constructs (HOC)**
- Added support for repeated-indicator and two-stage higher-order construct modeling.
- Added automatic HOC detection and model expansion utilities.
- Added HOC schema definitions and validation.

**Measurement Invariance (MICOM)**
- Added MICOM workflow for assessing measurement invariance across groups.
- Added invariance assessment outputs and reporting structures.

**Multi-Group Analysis (MGA)**
- Added MGA engine for comparing structural paths across groups.
- Added group comparison endpoints and result schemas.
- Added support for significance testing of path differences.

**Backend**
- New `app/engine_mga.py`.
- Added `/hoc` and `/mga` API endpoints.
- Extended parser and schema support for HOC, MICOM and MGA workflows.

**Frontend**
- Major update to `static/index.html`.
- Added user interface support for HOC and multi-group analysis workflows.
- Improved results presentation and reporting experience.

### Infrastructure
- Added `.gitattributes` for consistent line-ending handling across platforms.

### Changed
- Version bumped to `0.6.0`.


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

*v1.0.0 completes the dependency-ordered feature roadmap (v0.3 → v1.0).
See [NAVAL-SEM Release Plan](docs/) for the full feature history and sizing notes.*

