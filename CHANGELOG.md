# NAVAL-SEM — Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [2.0.0] - 2026-08-22

### ⚠ BREAKING

- **`POST /nomological` response shape changed from a bare JSON array to a wrapped object.** Was `List[NomologicalResult]` (e.g. `[{"construct": "Y", "r_squared": 0.42, ...}]`); now `NomologicalBatchResult` (`{"entries": [...], "warnings": [...], "fingerprint": "...", "anchor_status": null}`). **This does not change any analysis output** — `compute_nomological_validity()` and the per-item `NomologicalResult` fields (`construct`, `r_squared`, `benchmark`, `passed`, `interpretation`) are byte-for-byte unchanged; the numbers you get are identical. What breaks is purely mechanical: any caller doing `response[0].r_squared` or `response.map(...)` needs to change to `response.entries[0].r_squared` / `response.entries.map(...)`, since the top-level JSON is now an object instead of an array. Necessary because a bare array cannot carry a top-level `fingerprint` — see the fingerprint/provenance entry below.

### Added

- **Fingerprint + optional Bitcoin timestamp extended to every result-producing analysis endpoint, not just `/run`.** Previously only the main SEM fit (`/run`) and its `.docx`/PDF exports carried a reproducibility fingerprint — all ~22 other statistical methods (MGA, HOC, moderation, IPMA, NCA, NCA-ESSE, fsQCA, robustness checks, FIMIX, PLS-POS, LCA, moderated mediation, nomological validity, measurement invariance, CTA, multi-group CB-SEM, EFA, CVI, Bayesian SEM, bootstrap, HTMT, predictive relevance, CMB, indirect effects) had none, even though many of these produce results just as likely to be independently cited in a paper as the main model fit. Extended provenance to all of them:
  - New `_compute_fingerprint_generic()` in `app/main.py` — a reusable fingerprinting function not tied to SEM-specific fields (CFI/RMSEA/etc.), so it works for fundamentally different result shapes (fsQCA solutions, LCA class assignments, CTA tetrad tests, etc.). The original SEM-specific `_compute_fingerprint()` (used by `/run` and `/export/docx`) is now a thin wrapper around it, preserving its existing audit shape.
  - New `_attach_provenance()` helper shared by every endpoint: computes the fingerprint, optionally submits it for Bitcoin timestamping (only if the endpoint's new `anchor` flag is `True` — off by default everywhere), attaches `fingerprint`/`anchor_status` onto the result, and registers it in `_run_store` so `/fingerprint/{run_id}`, `/proof`, and `/upgrade` work uniformly across all endpoints, not just `/run`.
  - `fingerprint: Optional[str]` and `anchor_status: Optional[str]` fields added to all 22 corresponding response schemas in `app/schemas.py`.
  - Every affected endpoint gained a `run_id: str = Form(None)` (where not already present) and `anchor: bool = Form(False)` parameter, matching `/run`'s existing pattern.
  - **`/nomological` required a breaking response-shape change**: it previously returned a bare JSON array (`List[NomologicalResult]`), which cannot carry a top-level fingerprint field. Introduced `NomologicalBatchResult` (`{entries, warnings, fingerprint, anchor_status}`) as its new response model. This is non-breaking in practice for this codebase's own frontend — `_renderNomologicalResults()` already defensively handled both a bare array and a `{entries: [...]}` object (apparently anticipating this exact change before the backend ever sent it), so no frontend logic needed to change beyond simplifying a now-dead code comment and adding a fingerprint display line.
  - All of the above degrade gracefully exactly like `/run`: `anchor=True` with no internet/calendar access still returns the full analysis result normally, with `anchor_status` reporting `timeout`/`failed` rather than the request failing.
  - Tested end-to-end (not just compiled) for a representative cross-section covering every structural variant present in the 22 endpoints: standard file+Form (`/predict`, `/mga`), no pre-existing `run_id` (`/htmt`), JSON-body request (`/bayesian-sem`), no `run_id`/log infrastructure at all pre-change (`/fsqca`, `/cvi`), and the array-to-object breaking change (`/nomological`).

- **Fingerprint (and Bitcoin timestamp, if requested) now travels with every exported artifact, not just the in-app Downloads tab.** Previously the hash only existed inside the running app session — nothing that actually left the system (CSV, JSON, R/Python/lavaan code exports, the APA `.docx` report, the PDF report) carried it, which defeated the point of timestamping a result meant to be cited in a paper: a reader of the published output had no way to find the hash to verify against. Now:
  - Every CSV export gets a trailing `"Provenance",...` section with the fingerprint and Bitcoin timestamp status.
  - The JSON export (`validity-json`) gets a `provenance` object with the same fields plus a one-line note on how to verify.
  - R/Python/lavaan code exports get a `#`-comment header with the same info.
  - The APA `.docx` report gets a new **"Reproducibility & Provenance"** section right after the title, spelling out what the fingerprint covers, the exact Bitcoin timestamp status, and — precisely, since this matters for a research audience — that only the hash (never data/syntax/results) ever left the machine, with no wallet, mining, or payment performed by NAVAL-SEM.
  - The PDF report gets the equivalent section (`_build_provenance_section` in `app/export_pdf.py`), sourced directly from the `results` payload the frontend already sends.
  - `/export/docx` gained its own `anchor: bool = Form(False)` flag (default off, matching `/run`) and now computes its own fresh fingerprint over the exact re-fitted model in that export — the docx export doesn't reuse the interactive session's fingerprint, since its own `bootstrap_n`/`reverse_items` can differ from that session's, and the hash embedded in the document must match what's actually in the document.
  - All of the above degrade gracefully: exports with no fingerprint (e.g. an older client, or a payload that predates this feature) simply omit the provenance section/fields entirely rather than showing an empty or broken block.

### Fixed

- **`opentimestamps-client` crashed on import on Windows.** The `ots` CLI's `cmds.py` unconditionally imports `bitcoin.rpc`/`bitcoin.wallet` (needed only for its local Bitcoin node RPC mode, which was never used here), and those modules call `ctypes.cdll.LoadLibrary` for OpenSSL at import time — which fails with `TypeError: argument of type 'NoneType' is not iterable` on Windows machines without `libssl`/`libcrypto` on the DLL search path. Rewrote `app/anchor.py` to talk to public OpenTimestamps calendar servers directly via the underlying `opentimestamps` core library (`opentimestamps.calendar.RemoteCalendar`, `opentimestamps.core.timestamp`) instead of shelling out to the `ots` CLI. This only imports `bitcoin.core` (safe — no ctypes/OpenSSL involved), never `bitcoin.rpc`/`bitcoin.wallet`. Swapped the `opentimestamps-client==0.7.2` dependency for `opentimestamps==0.4.5` (the CLI wrapper is no longer needed at all). Verified the app runs correctly with `opentimestamps-client` fully uninstalled.

### Changed

- **Clarified "anchor" terminology.** "Anchor fingerprint (Bitcoin)" implied NAVAL-SEM performs blockchain/mining work itself. It doesn't — it submits only a hash to free OpenTimestamps calendar servers, which batch many users' hashes into one Bitcoin transaction that someone else pays the fee for (a decentralized analogue of RFC 3161 trusted timestamping). Relabeled the checkbox and all related UI/log copy to **"Timestamp on Bitcoin (via OpenTimestamps, free)"** and expanded the tooltip to spell out the actual mechanism, since precision matters more for the academic (PhD/researcher) audience this feature targets than for a general consumer app.
- **Made explicit that Bitcoin timestamping is off by default and the proof does not persist server-side.** No blockchain/calendar server is ever contacted unless the "Timestamp on Bitcoin" box is checked before a run — this is stated directly in the checkbox tooltip, the fingerprint panel, and the Downloads-tab provenance card, not just implied by the checkbox being unchecked. Also made explicit that the `.ots` proof only lives in server memory (`_run_store`) for the current session; it is not written to disk, so it must be downloaded promptly after a run (or before restarting the server) or it is lost and the run would need to be repeated to get a new one.

### Added

- **Optional Bitcoin timestamping of run fingerprints (OpenTimestamps)** — Every run already computed a local SHA-256 "fingerprint" covering model syntax, data hash, algorithm, environment, and key fit results (`_compute_fingerprint`), but that fingerprint carried no independent proof of *when* it was produced. Added an opt-in `anchor` flag on `POST /run` (default `false`, so the app stays fully offline-capable by default) which, when set, submits the fingerprint hash to free public OpenTimestamps calendar servers — no wallet, no funds, no on-chain transaction constructed or signed by NAVAL-SEM itself, and no mining performed by this app. Returns a `.ots` proof (`GET /fingerprint/{run_id}/proof`) that anyone can independently verify once the underlying Bitcoin transaction confirms (usually within hours); `POST /fingerprint/{run_id}/upgrade` re-checks and finalizes the proof. Intended for results that will be cited in papers/journals, as independent, trustless corroboration alongside the project's existing Zenodo DOI. Also fixed the frontend, which previously displayed "Fingerprint anchored" for *every* run even though no anchoring had ever occurred — it now correctly reads "Fingerprint computed locally" by default, with a genuine status message only shown when timestamping was requested. Affects `app/anchor.py` (new), `app/main.py`, `app/schemas.py`, `static/index.html`, `pyproject.toml`.

### Added

- **Fingerprint + optional Bitcoin timestamp extended to every result-producing analysis endpoint, not just `/run`.** Previously only the main SEM fit (`/run`) and its `.docx`/PDF exports carried a reproducibility fingerprint — all ~22 other statistical methods (MGA, HOC, moderation, IPMA, NCA, NCA-ESSE, fsQCA, robustness checks, FIMIX, PLS-POS, LCA, moderated mediation, nomological validity, measurement invariance, CTA, multi-group CB-SEM, EFA, CVI, Bayesian SEM, bootstrap, HTMT, predictive relevance, CMB, indirect effects) had none, even though many of these produce results just as likely to be independently cited in a paper as the main model fit. Extended provenance to all of them:
  - New `_compute_fingerprint_generic()` in `app/main.py` — a reusable fingerprinting function not tied to SEM-specific fields (CFI/RMSEA/etc.), so it works for fundamentally different result shapes (fsQCA solutions, LCA class assignments, CTA tetrad tests, etc.). The original SEM-specific `_compute_fingerprint()` (used by `/run` and `/export/docx`) is now a thin wrapper around it, preserving its existing audit shape.
  - New `_attach_provenance()` helper shared by every endpoint: computes the fingerprint, optionally submits it for Bitcoin timestamping (only if the endpoint's new `anchor` flag is `True` — off by default everywhere), attaches `fingerprint`/`anchor_status` onto the result, and registers it in `_run_store` so `/fingerprint/{run_id}`, `/proof`, and `/upgrade` work uniformly across all endpoints, not just `/run`.
  - `fingerprint: Optional[str]` and `anchor_status: Optional[str]` fields added to all 22 corresponding response schemas in `app/schemas.py`.
  - Every affected endpoint gained a `run_id: str = Form(None)` (where not already present) and `anchor: bool = Form(False)` parameter, matching `/run`'s existing pattern.
  - **`/nomological` required a breaking response-shape change**: it previously returned a bare JSON array (`List[NomologicalResult]`), which cannot carry a top-level fingerprint field. Introduced `NomologicalBatchResult` (`{entries, warnings, fingerprint, anchor_status}`) as its new response model. This is non-breaking in practice for this codebase's own frontend — `_renderNomologicalResults()` already defensively handled both a bare array and a `{entries: [...]}` object (apparently anticipating this exact change before the backend ever sent it), so no frontend logic needed to change beyond simplifying a now-dead code comment and adding a fingerprint display line.
  - All of the above degrade gracefully exactly like `/run`: `anchor=True` with no internet/calendar access still returns the full analysis result normally, with `anchor_status` reporting `timeout`/`failed` rather than the request failing.
  - Tested end-to-end (not just compiled) for a representative cross-section covering every structural variant present in the 22 endpoints: standard file+Form (`/predict`, `/mga`), no pre-existing `run_id` (`/htmt`), JSON-body request (`/bayesian-sem`), no `run_id`/log infrastructure at all pre-change (`/fsqca`, `/cvi`), and the array-to-object breaking change (`/nomological`).

- **Fingerprint (and Bitcoin timestamp, if requested) now travels with every exported artifact, not just the in-app Downloads tab.** Previously the hash only existed inside the running app session — nothing that actually left the system (CSV, JSON, R/Python/lavaan code exports, the APA `.docx` report, the PDF report) carried it, which defeated the point of timestamping a result meant to be cited in a paper: a reader of the published output had no way to find the hash to verify against. Now:
  - Every CSV export gets a trailing `"Provenance",...` section with the fingerprint and Bitcoin timestamp status.
  - The JSON export (`validity-json`) gets a `provenance` object with the same fields plus a one-line note on how to verify.
  - R/Python/lavaan code exports get a `#`-comment header with the same info.
  - The APA `.docx` report gets a new **"Reproducibility & Provenance"** section right after the title, spelling out what the fingerprint covers, the exact Bitcoin timestamp status, and — precisely, since this matters for a research audience — that only the hash (never data/syntax/results) ever left the machine, with no wallet, mining, or payment performed by NAVAL-SEM.
  - The PDF report gets the equivalent section (`_build_provenance_section` in `app/export_pdf.py`), sourced directly from the `results` payload the frontend already sends.
  - `/export/docx` gained its own `anchor: bool = Form(False)` flag (default off, matching `/run`) and now computes its own fresh fingerprint over the exact re-fitted model in that export — the docx export doesn't reuse the interactive session's fingerprint, since its own `bootstrap_n`/`reverse_items` can differ from that session's, and the hash embedded in the document must match what's actually in the document.
  - All of the above degrade gracefully: exports with no fingerprint (e.g. an older client, or a payload that predates this feature) simply omit the provenance section/fields entirely rather than showing an empty or broken block.

### Fixed

- **`opentimestamps-client` crashed on import on Windows.** The `ots` CLI's `cmds.py` unconditionally imports `bitcoin.rpc`/`bitcoin.wallet` (needed only for its local Bitcoin node RPC mode, which was never used here), and those modules call `ctypes.cdll.LoadLibrary` for OpenSSL at import time — which fails with `TypeError: argument of type 'NoneType' is not iterable` on Windows machines without `libssl`/`libcrypto` on the DLL search path. Rewrote `app/anchor.py` to talk to public OpenTimestamps calendar servers directly via the underlying `opentimestamps` core library (`opentimestamps.calendar.RemoteCalendar`, `opentimestamps.core.timestamp`) instead of shelling out to the `ots` CLI. This only imports `bitcoin.core` (safe — no ctypes/OpenSSL involved), never `bitcoin.rpc`/`bitcoin.wallet`. Swapped the `opentimestamps-client==0.7.2` dependency for `opentimestamps==0.4.5` (the CLI wrapper is no longer needed at all). Verified the app runs correctly with `opentimestamps-client` fully uninstalled.

### Changed

- **Clarified "anchor" terminology.** "Anchor fingerprint (Bitcoin)" implied NAVAL-SEM performs blockchain/mining work itself. It doesn't — it submits only a hash to free OpenTimestamps calendar servers, which batch many users' hashes into one Bitcoin transaction that someone else pays the fee for (a decentralized analogue of RFC 3161 trusted timestamping). Relabeled the checkbox and all related UI/log copy to **"Timestamp on Bitcoin (via OpenTimestamps, free)"** and expanded the tooltip to spell out the actual mechanism, since precision matters more for the academic (PhD/researcher) audience this feature targets than for a general consumer app.
- **Made explicit that Bitcoin timestamping is off by default and the proof does not persist server-side.** No blockchain/calendar server is ever contacted unless the "Timestamp on Bitcoin" box is checked before a run — this is stated directly in the checkbox tooltip, the fingerprint panel, and the Downloads-tab provenance card, not just implied by the checkbox being unchecked. Also made explicit that the `.ots` proof only lives in server memory (`_run_store`) for the current session; it is not written to disk, so it must be downloaded promptly after a run (or before restarting the server) or it is lost and the run would need to be repeated to get a new one.

### Added

- **Optional Bitcoin timestamping of run fingerprints (OpenTimestamps)** — Every run already computed a local SHA-256 "fingerprint" covering model syntax, data hash, algorithm, environment, and key fit results (`_compute_fingerprint`), but that fingerprint carried no independent proof of *when* it was produced. Added an opt-in `anchor` flag on `POST /run` (default `false`, so the app stays fully offline-capable by default) which, when set, submits the fingerprint hash to free public OpenTimestamps calendar servers — no wallet, no funds, no on-chain transaction constructed or signed by NAVAL-SEM itself, and no mining performed by this app. Returns a `.ots` proof (`GET /fingerprint/{run_id}/proof`) that anyone can independently verify once the underlying Bitcoin transaction confirms (usually within hours); `POST /fingerprint/{run_id}/upgrade` re-checks and finalizes the proof. Intended for results that will be cited in papers/journals, as independent, trustless corroboration alongside the project's existing Zenodo DOI. Also fixed the frontend, which previously displayed "Fingerprint anchored" for *every* run even though no anchoring had ever occurred — it now correctly reads "Fingerprint computed locally" by default, with a genuine status message only shown when timestamping was requested. Affects `app/anchor.py` (new), `app/main.py`, `app/schemas.py`, `static/index.html`, `pyproject.toml`.

---

## [v1.1.11] — 2026-08-15 · macOS Build and previous [v1.1.10] HOC Fix 

### Fixed

- **HOC guidance in Main Run never actually reached the user (v1.1.9 follow-up)** — v1.1.9 added HOC syntax detection and a guidance message for Main Run, but two bugs meant it never fired for the case it was meant to catch:
  - A model with an un-expanded HOC block (e.g. `HOC =~ FOC1 + FOC2`) always fails `/run` — `FOC1`/`FOC2` are latent-variable names, not dataset columns, so `fit_model()` rejects them as missing columns. The v1.1.9 HOC check only ran *after a successful run*, so for the one case it existed for, it was unreachable. The failure itself was written only to `#canvas-error` on the Model tab, which is invisible while looking at Analyse → Main run — so the panel there just kept showing a stale "No results yet." with no sign anything had happened. Reported by a user whose customer thought the app was broken.
  - `detectHOCFromSyntax()` treated *any* structural path between two ordinary latent variables (`Y ~ X`) as if it were a HOC signal, alongside genuine `HOC =~ FOC1 + FOC2`-style measurement blocks. That's normal SEM, not a HOC — the bug would have produced a false "your model contains HOC" warning on most ordinary multi-construct models the moment it was surfaced anywhere visible.

  Fixed by rewriting the detector to match the backend's own `detect_hoc()` (measurement/formative blocks only), and by having both `#core-res-body` (Analyse → Main run) and `#res-body` (Model tab) actually re-render on a failed run instead of only touching `#canvas-error`. A failed run whose missing columns are explained by an un-expanded HOC now shows which construct(s) are involved and a **Go to HOC tab** button that jumps straight there; other failures show the real error instead of a silent "no results yet". `fit_model()` also now raises a specific, actionable message for this case (rather than a generic "columns not found") so it's clear from the `/run` response directly, independent of the UI.

  Affects `app/engine.py`, `static/index.html`.

- **macOS build was missing a local entry point and wasn't using a checked-in script in CI** — `docs/building.md` documented `./build_macos.sh` as the local macOS build command, but the script didn't exist in the repo. Meanwhile the `build-macos` job in `.github/workflows/release.yml` duplicated the same PyInstaller/`create-dmg` packaging logic directly inline instead of delegating to a script, unlike the Linux job, which already ran `build_linux.sh`. Added `build_macos.sh` (builds `dist/NAVAL-SEM.app` via PyInstaller, falling back to a direct `--windowed` build since `naval_sem.spec` only defines an `EXE()` target and never produces a `.app` bundle on its own, then packages `dist/NAVAL-SEM.dmg` with `create-dmg`), and pointed the CI job at it so local and CI builds now share one source of truth. Affects `build_macos.sh` (new), `.github/workflows/release.yml`.

---
## [v1.1.9] — 2026-08-11 · UI/UX Improvements

### Added

- **HOC syntax auto-detection and user guidance** — When a model contains Higher-Order Constructs (HOC) and the user runs Main Run, the app now detects this from the syntax and displays a helpful warning message directing the user to the Analyse → HOC tab. The HOC tab in the results drawer is also automatically unhidden when HOC is detected, making it easier for users to find the correct analysis method. Detection works for both measurement model syntax (`HOC =~ FOC1 + FOC2`) and structural model syntax (`TL == IB + IA + IM`).

### Changed

- **Downloads tab label** — The confusing download arrow icon (⬇) in both the results drawer and core results panel has been replaced with the clearer text label "Outputs" to make it obvious what the tab contains.

### Fixed

- **User confusion with HOC models** — Previously, users with HOC syntax who ran Main Run would get no output without understanding why. Now they receive clear guidance to use the HOC analysis tab instead.

---
## [v1.1.8] — 2026-08-07 · Better UI
## [v1.1.6] — 2026-08-07 · Column Pill Scroll Fix

### Fixed

-**Dataset column pills didn't scroll horizontally and instead overflowed the upload bar**. #col-pills-wrap (Model tab), #mga-col-pills (MGA tab), and #hoc-col-pills (HOC tab) are each a flex: 1 item inside a display: flex parent (.upbar) with overflow-x: auto set directly on them. Flex items default to min-width: auto, which floors an item's width at its own content size regardless of flex or overflow settings — so the pill row was never allowed to shrink to the container's width, and the intended internal scroll never had anything to scroll within; it just spilled past the visible edge of the bar instead. Added min-width: 0 to all three containers so they now shrink to the available space and scroll as intended. Affects static/index.html.

---

## [v1.1.6] — 2026-08-07 · Column Pill Scroll Fix

### Fixed
- **Dataset column pills didn't scroll horizontally and instead overflowed the upload bar.** `#col-pills-wrap` (Model tab), `#mga-col-pills` (MGA tab), and `#hoc-col-pills` (HOC tab) are each a `flex: 1` item inside a `display: flex` parent (`.upbar`) with `overflow-x: auto` set directly on them. Flex items default to `min-width: auto`, which floors an item's width at its own content size regardless of `flex` or `overflow` settings — so the pill row was never allowed to shrink to the container's width, and the intended internal scroll never had anything to scroll within; it just spilled past the visible edge of the bar instead. Added `min-width: 0` to all three containers so they now shrink to the available space and scroll as intended.
- **Even after the row could scroll, a plain vertical mouse wheel still did nothing.** `overflow-x: auto` only responds natively to Shift+wheel or a trackpad's horizontal swipe gesture — no code translated an ordinary vertical wheel scroll into horizontal movement, so the row was still practically unreachable with a standard mouse. Added a `wheel` listener on all three containers that scrolls them horizontally on plain vertical wheel input (skipped when the row has nothing to scroll, or when the input is already horizontal). Also added a right-edge fade (`.col-pill-scroll`, matching the existing `.model-pill` pattern) that clears on hover, since the scrollbar itself is intentionally hidden (`scrollbar-width: none`) and gave no visual cue that more columns existed.

Affects `static/index.html`.

---

## [v1.1.5] — 2026-08-02 · Report Diagram Theme-Refresh Fix

### Fixed
- **Exported report diagrams could show a stale theme even after switching themes.** `buildReport()` (which generates the diagram, among the rest of the report) only runs when the Report tab is opened or reopened — `downloadReport()` deliberately skips rebuilding the whole card to preserve in-place edits (report note, metadata). If a theme was applied *after* the Report tab was last built, the diagram markup already sitting in the DOM still had the previous theme's colors baked directly into its SVG `fill`/`stroke`/`background` attributes, so no export-time backgroundColor option could correct it — the v1.1.4 html2canvas fix was correct but never got exercised, since the SVG itself was already wrong before capture. `downloadReport()` now regenerates the diagram from `_buildSVGFromBuilderNodes()` (which reads the live `window._CT` theme object) immediately before capturing, so the export always reflects whichever theme is active at that moment, regardless of when the Report tab was last opened. Affects `static/index.html`.
- **A second, independent copy of the v1.1.4 HOC-blind rank-layout bug**, inside `_buildSVGFromBuilderNodes()`'s overlap-recovery path (triggers only when a loaded snapshot's nodes are already overlapping, e.g. a `results.json` saved before v1.1.4). Given the same HOC-aware rank fix as the primary layout algorithm, so it can't silently reintroduce the single-column stacking bug via this fallback path. Affects `static/index.html`.

---

## [v1.1.4] — 2026-08-02 · Diagram Layout & Export Theme Fix

### Fixed
- **HOC model canvas auto-layout stacked every construct into a single overlapping column.** `_topoLayout()` computed each latent variable's column position by walking only `structural` (`~`) edges. A higher-order construct spec such as `g =~ visual + textual + speed` uses a measurement (`=~`) edge, not a structural one, so every LV's rank stayed at 0 and `visual`/`textual`/`speed`/`g` were all placed in the same column, on top of each other. LV-to-LV measurement edges (HOC relationships) are now folded into the rank computation, reversed in direction so first-order constructs rank before the higher-order construct they feed into. Affects `static/index.html`.
- **Indicator-clustering step could misfile a first-order construct as an "indicator" of its own higher-order construct.** The same layout pass re-clusters observed indicators (`x1`, `x2`, ...) around their construct using measurement edges, but didn't distinguish an LV-to-LV edge (e.g. `g =~ visual`) from a genuine indicator edge (e.g. `visual =~ x1`) — so `visual` itself could get pulled into a small indicator slot next to `g`. LV-to-LV edges are now excluded from this step. Affects `static/index.html`.
- **Report/diagram exports always rendered with a hardcoded dark background (`#13151a`), regardless of the active theme.** Both the PNG export and the PDF path-diagram rasterization in `downloadReport()` ignored whatever theme (including a custom-uploaded light theme) was active in the UI. Both now read the live `--bg` CSS variable at export time, falling back to the previous dark value only if it isn't set. Affects `static/index.html`.

### Removed
- **Dead force-simulation code.** `_d3sim` (a `d3.forceSimulation` with link/charge/collide forces and a tick handler) was declared, configured, and immediately stopped, but never given nodes/links or restarted anywhere — so it never ran. Removed; the actual auto-layout is the rank/column algorithm above. Affects `static/index.html`.

---

## [v1.1.3] — 2026-08-01 · Bootstrap Back-Fill Fix

### Added

**Formative construct (Mode B) support — PLS-SEM**
- Added `<~` lavaan-style operator for formative measurement blocks (e.g. `Quality <~ q1 + q2 + q3`), alongside the existing `=~` reflective operator. Checked before `=~` in `parse_lavaan()`'s operator chain so `<~` lines aren't swallowed by the plain `~` branch; `preprocess_lavaan()` now also treats a trailing `<~` as a line-continuation marker for multi-line formative blocks.
- `parse_lavaan()` returns two new keys: `formative_lvs` (LVs declared with `<~`) and `construct_modes` (`{lv: "A"|"B"}` for every LV in the model, defaulting to `"A"` when a construct has no formative declaration).
- `PLSEstimator`'s outer-weight update now branches per construct on `construct_modes`: Mode B blocks solve `w = (X'X + ridge·I)⁻¹ X'η_inner` (ridge = 1e-4, falling back to the unregularised solve only if that raises `LinAlgError`), normalised so `Var(η) = 1`; Mode A blocks are unchanged. `PLSResult` gains a `construct_modes` field carrying the per-LV mode through to the response.
- Affects `app/parser.py`, `app/pls.py`.

### Changed

- **`compute_vif()` now skips Mode A (reflective) constructs** — VIF/multicollinearity is only a meaningful diagnostic for formative blocks; reflective indicators are expected to correlate highly since they share a common cause. Models composed entirely of reflective constructs now return an empty VIF list instead of one entry per indicator. Affects `app/engine.py`.

### Fixed

- **HOC / measurement (`=~`) loadings never received bootstrap SE, z, or CI when `bootstrap_n > 0`** — the bootstrap significance back-fill in `_run_diagnostics()` previously only scanned structural (`~`) parameters when deciding whether a back-fill was needed, and only wrote `significant`/`p_value`/CI onto matched parameters. Measurement loadings — including higher-order construct loadings produced by the repeated-indicator HOC expansion (e.g. `x1 =~ g`) — were never flagged for back-fill and kept the PLS point-estimate placeholders (`std_error=0.0`, `z_value=0.0`, `p_value=1.0`) in every report regardless of `bootstrap_n`. The trigger condition and the back-fill body now cover any hypothesis-tested parameter (`op` in `{"~", "=~"}`; `"~~"` covariance rows stay excluded, since they are never hypothesis-tested), and `std_error`/`z_value` are populated from the bootstrap resampling distribution alongside significance and CI bounds. Affects `app/engine.py`.
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
- Version bumped to 0.7.0.
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

