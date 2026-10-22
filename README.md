# NAVAL-SEM

**Free, offline PLS-SEM · CB-SEM · fsQCA desktop application.**
Visual model builder, bootstrapping, HTMT, APA 7th-edition reporting, and R/Python export — no internet required, no licence, no sample size limit.

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Latest Release](https://img.shields.io/github/v/release/navalsingh9/naval-sem)](https://github.com/navalsingh9/naval-sem/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)](https://github.com/navalsingh9/naval-sem/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20124108.svg)](https://doi.org/10.5281/zenodo.20124108)
[![Discussions](https://img.shields.io/github/discussions/navalsingh9/naval-sem)](https://github.com/navalsingh9/naval-sem/discussions)

<p align="center">
  <img src="https://github.com/navalsingh9/naval-sem/blob/master/docs/community/badges/oss-rising-star-black.png" width="120" alt="SourceForge Rising Star Award">
  <br>
  <sub><b>SourceForge Rising Star Award (2026)</b></sub>
</p>

---

## Get it

<a href="https://sourceforge.net/projects/naval-sem/files/latest/download" target="_blank">
  <img src="https://a.fsdn.com/con/app/sf-download-button" alt="Download NAVAL-SEM">
</a>

Fastest way to get the newest stable release. Mirrors: [SourceForge](https://sourceforge.net/projects/naval-sem/) · [GitHub Releases](https://github.com/navalsingh9/naval-sem/releases/latest)

| Platform | File | Notes |
|---|---|---|
| Windows | `NAVAL-SEM-Setup.msi` | Recommended — Start Menu integration + uninstall |
| Windows | `NAVAL-SEM.exe` | Portable — no installation required |
| macOS (Apple Silicon) | `NAVAL-SEM-arm64.dmg` | M1/M2/M3/M4 — drag into Applications |
| macOS (Intel) | `NAVAL-SEM-x86_64.dmg` | Intel Macs — drag into Applications |
| Linux | `naval-sem_*.deb` | Debian / Ubuntu installer |
| Linux | `NAVAL-SEM` | Portable binary |

> **Windows:** SmartScreen may appear on first launch — *More info → Run anyway*.
> **macOS:** Right-click → *Open* on first launch to bypass Gatekeeper for unsigned apps.

---

## What's new in v2.0.11

**fsQCA exports now render consistently in PNG and PDF.** Both output formats now include the same necessity and sufficiency information, clearly identify fsQCA results, and omit irrelevant SEM metrics so the generated reports stay faithful to the chosen analysis.

Also in this release:
- Fixed the mismatch between exported fsQCA report content and the report window
- Kept PDF and PNG exports aligned for necessity tables and solution summaries
- Preserved the app’s existing 2.0.x compatibility and release versioning across project metadata

<sub>Full history in [Changelog](#changelog).</sub>

---

## Why NAVAL-SEM

Built for people who need real SEM output without a subscription or a data-privacy trade-off:

- **PhD students** who need full PLS-SEM without a SmartPLS licence
- **Professors** who want a free, zero-install classroom tool
- **HR, marketing, and healthcare practitioners** running structural diagnostics on sensitive data that can't leave the machine
- **Anyone** who wants SmartPLS-quality output without the SmartPLS price

NAVAL-SEM starts a local FastAPI server on `127.0.0.1:8765` and opens the UI in a native window. Nothing is transmitted anywhere unless you explicitly opt into timestamping a fingerprint hash — and even then only a hash leaves the machine.

**v2.0.11** is the current stable release. **v1.1.5 LTS** (2 Aug 2026) remains the long-term-support milestone with frozen public API schemas — cite this build in your methods section if schema consistency matters for reproducibility.

---

## Features

<table>
<tr><td valign="top" width="50%">

**Core SEM**
- PLS-SEM — reflective + formative, 5,000-iteration bootstrap, indirect effects, mediation classification
- CB-SEM — lavaan syntax, CFI/RMSEA/SRMR/χ²-df
- WLS estimator
- Visual canvas — drag-and-drop, live validity warnings, Undo/Redo, PNG export
- AVE, Composite Reliability, Cronbach's α, cross-loadings, Fornell-Larcker
- HTMT with configurable threshold (0.85 / 0.90)

**Effects & validity**
- VIF (strict <3.3 / acceptable <5.0)
- Cohen's f² per structural path
- Bootstrapped mediation, 95% CI
- Q² blindfolding, PLSpredict (k-fold), CVPAT
- Common Method Bias (Lindell & Whitney 2001)

</td><td valign="top" width="50%">

**Advanced analysis**
- Higher-Order Constructs — repeated-indicator + two-stage
- Multi-Group Analysis — permutation significance
- Moderation & moderated mediation — Hayes PROCESS 7, 14, 58/59
- IPMA, NCA (CE-FDH / CR-FDH), FIMIX-PLS, PLS-POS

**Scale development**
- CVI (Polit & Beck 2006), EFA, nomological validity
- Measurement invariance (MICOM extended)
- NCA-ESSE (Becker, Richter, Ringle & Sarstedt 2026)

**fsQCA & reporting**
- Direct calibration, truth table, Quine-McCluskey minimisation
- Complex / parsimonious / intermediate solutions, coincidence bubble chart
- One-click APA 7 `.docx` export, PDF report export
- R (lavaan/seminr), Python (semopy), JASP code export

</td></tr>
</table>

---

## NAVAL-SEM vs SmartPLS

| | NAVAL-SEM v2.0 | SmartPLS 4 |
|---|---|---|
| Price | **Free** | Paid licence |
| Sample size / construct limit | **No application-imposed limit** | Plan-dependent; verify current SmartPLS terms |
| OS | **Windows · macOS · Linux** | Windows · macOS only |
| Offline | **Always** | After licence activation |
| CB-SEM | **ML + WLS estimators** | Limited |
| fsQCA | **Quine-McCluskey minimisation** | Not compared here; verify current SmartPLS documentation |
| APA 7 reporting | **One-click `.docx`** | Not compared here; verify current SmartPLS documentation |
| R / Python export | **lavaan, seminr, semopy** | Not compared here; verify current SmartPLS documentation |
| Result provenance | **Optional Bitcoin timestamping** | Not compared here; verify current SmartPLS documentation |
| Data privacy | **Localhost only** | Local analysis |
| Schema stability | **Frozen public API (LTS)** | Proprietary |
| Citable DOI | **Zenodo, every release** | Not applicable as a project citation claim |

---

## SEM Case Library

Five production-ready research cases, each with a real open dataset, construct specification, and expected path coefficients — load one to start a working model immediately.

| # | Case | Framework | Dataset | n | Key finding |
|---|---|---|---|---|---|
| 01 | Why Employees Really Quit | JD-R Theory | IBM HR Attrition · Kaggle | 1,470 | WLB→Intent β=0.08 n.s. once Manager Trust enters — full mediation |
| 02 | Why Awareness Doesn't Become Purchase | Aaker Brand Equity | Customer Personality · Kaggle | 2,216 | Quality→Trust β=0.19 — the funnel gap regression misses |
| 03 | Why Hospital Ratings Fall | Modified SERVQUAL | HCAHPS · CMS.gov | ~4,800 | Responsiveness→Overall β=0.11 n.s. — it's communication, not call speed |

→ [Browse `cases/`](https://github.com/navalsingh9/naval-sem/tree/master/cases)

---

## Provenance & integrity

NAVAL-SEM can optionally anchor a **fingerprint** (a SHA-256 hash of the analysis inputs and outputs) to the Bitcoin blockchain through the free, public [OpenTimestamps](https://opentimestamps.org/) calendar network. This is entirely opt-in and off by default. It is useful as a tamper-evident, dated record of an analysis—for example, for a thesis or data-integrity record, resolving a dispute about when results were produced, or documenting that results existed before a preprint or submission.

### How it works

1. Before starting an analysis, check **₿ Timestamp** in the top bar. The option applies to the analysis methods in the application; if it is left unchecked, no blockchain or OpenTimestamps request is made.
2. NAVAL-SEM computes the fingerprint locally. Only that hash is submitted to free public OpenTimestamps calendar servers; your dataset, model, and results stay on your machine.
3. The run panel shows the timestamp status and provides **Download `.ots` Proof**. Download and retain this proof with the corresponding exported result or audit JSON—the server does not store it permanently.
4. The initial status is normally **Pending Bitcoin confirmation**. Confirmation is asynchronous and usually takes a few hours to about a day. Use **Check confirmation** later; once confirmed, the proof is self-contained and can be independently verified without trusting NAVAL-SEM.

The fingerprint and anchor status are embedded in every export format: CSV, JSON, R/Python/lavaan code, the APA `.docx` report, and the PDF report. Standalone reports (e.g. a fsQCA-only PDF) carry the same provenance metadata when generated from a valid run.

### What a timestamp does—and does not—establish

- **Does:** provide an independently verifiable record that this particular fingerprint, and therefore this particular analysis record, existed by the anchored time.
- **Does not:** prove authorship, originality, novelty, priority, or that the underlying data are genuine or were collected correctly. It is not a substitute for publication, a thesis submission, or patent filing. In most patent systems, priority depends on filing requirements rather than a blockchain timestamp; consult a qualified intellectual-property professional for advice about your situation.
- **Does not:** involve buying, holding, trading, or mining any cryptocurrency. No wallet, funds, or account are required.
- **Does not:** transmit your data—only a hash leaves your machine, never the underlying dataset.

Keep the `.ots` proof and the matching fingerprint/export together. Verification depends on preserving that proof and having access to the OpenTimestamps/Bitcoin verification ecosystem; “verifiable by anyone” does not mean that NAVAL-SEM or the proof file can be recovered if you lose them.

```
launcher.py
  ├── starts FastAPI on port 8765
  ├── opens pywebview window   →  http://127.0.0.1:8765
  ├── app/main.py              ← FastAPI app and route layer
  ├── app/engine.py            ← PLS/CB-SEM, bootstrapping, HTMT, MGA
  ├── app/fsqca.py             ← fsQCA engine
  ├── app/anchor.py            ← optional fingerprint + OpenTimestamps anchoring
  ├── app/export_docx.py       ← APA 7th-edition DOCX export
  ├── app/export_pdf.py        ← PDF report export
  ├── app/parser.py            ← CSV / Excel / SPSS ingestion
  └── static/index.html        ← canvas builder, results panels, export
```

No telemetry. No account. Anchoring sends a hash and nothing else — see [What touches the network](#what-touches-the-network) for the complete list of outbound requests the application can make.

---

## Run from source

```bash
git clone https://github.com/navalsingh9/naval-sem.git
cd naval-sem

uv sync --locked      # creates .venv and installs the exact locked dependency set
uv run launcher.py    # opens at http://127.0.0.1:8765
```

No `uv`? [Install it](https://docs.astral.sh/uv/getting-started/installation/) first. For packaging releases, use `uv sync --locked --group build` to include the PyInstaller toolchain.

→ Full build instructions (EXE, DMG, `.deb`): [`docs/building.md`](docs/building.md)

---

## Validation

Every release is gated on **174 pytest + Playwright tests**, compared against published anchor values from peer-reviewed literature. If the numbers don't match, the release is blocked.

- HS1939 CB-SEM CFI ≈ 0.931 (Holzinger & Swineford 1939)
- Bollen Political Democracy CFI ≥ 0.997 (Bollen 1989)
- Corporate Reputation avg loading ≈ 0.80, max HTMT ≈ 0.86 (Hair et al. 2011/2013)
- fsQCA consistency ≥ 0.80 (Wagemann & Schneider 2010; Ragin 2008)

Beyond those anchors, the suite checks FIML log-likelihood against Arbuckle (1996), fit indices against the `lavaan` tutorial's worked examples, Mardia's test on known normal and non-normal data, and repeated model checks to keep the app consistent with established SEM practice.

→ [Full test suite documentation](https://naval-sem.sourceforge.io/testbench.html)

**Requesting the suite.** The test code itself is not published, to limit derivative redistribution under the licence below. If you are evaluating NAVAL-SEM for research use and want to run the test suite, contact the project maintainer through the linked support channels.

---

## What touches the network

NAVAL-SEM performs **all analysis offline**. No dataset, variable name, model, or result ever leaves your machine, and every statistical feature works with networking disabled. Three optional features may contact the network only when enabled.

| Feature | When | Where it connects | What is sent |
|---|---|---|---|
| Update check | ~4 s after launch, and on demand via the toolbar button | `api.github.com` | Nothing but the request itself. Compares the latest release tag to your version. |
| Result anchoring | Only when you tick **anchor** on a run — off by default | OpenTimestamps calendars: `a.pool.opentimestamps.org`, `b.pool.opentimestamps.org`, `a.pool.eternitywall.com`, `btc.pool.opentimestamps.org` | Nothing but the fingerprint hash and the request metadata. |
| Theme fonts | Only if you pick a theme font that isn't one of the bundled DM Sans / DM Mono faces | `fonts.googleapis.com` | Nothing but the font request. |

To run fully air-gapped, decline the update check, leave anchoring off, and stay on the bundled fonts — or simply block the application at your firewall. Nothing degrades except those three features.

No telemetry. No analytics. No account. No data transmission of any kind beyond the three rows above.

---

## Citation

If you use NAVAL-SEM in published research, cite the project (this DOI always resolves to the latest release — no need to update it per version):

```
Singh, N. (2026). NAVAL-SEM: Free offline structural equation modelling
desktop application [Software]. https://doi.org/10.5281/zenodo.20124108
```

```bibtex
@software{singh2026navalsem,
  author  = {Singh, Naval},
  title   = {{NAVAL-SEM}: Free offline structural equation modelling desktop application},
  year    = {2026},
  doi     = {10.5281/zenodo.20124108},
  url     = {https://github.com/navalsingh9/naval-sem},
  license = {CC BY-NC-ND 4.0}
}
```

A `CITATION.cff` in the repository root supports APA 7, BibTeX, and RIS export.

For reproducibility, also record the exact NAVAL-SEM version used in your methods section. The DOI above resolves to the latest release; version-specific release records are available from [Zenodo](https://doi.org/10.5281/zenodo.20124108) and [GitHub Releases](https://github.com/navalsingh9/naval-sem/releases).

---

## Changelog

| Version | Date | Highlights |
|---|---|---|
| **v2.0.11** | 15 Oct 2026 | Fixed fsQCA report consistency across PNG/PDF exports; necessity + sufficiency outputs now align across report formats |
| **v2.0.10** | 17 Sep 2026 | Deterministic panel repaint flush — tool/tab active states now repaint even when WebKit refuses (OCLP/software renderers) |
| v2.0.9 | 17 Sep 2026 | Removed backdrop-filter from the tool/tab button hosts — .on states now repaint on any renderer, incl. OCLP Macs |
| v2.0.8 | 17 Sep 2026 | Fixed whole-window click freeze on legacy/OCLP Macs (surgical one-layer repaint, not mass promotion); empty-canvas probe skipped |
| v2.0.7 | 17 Sep 2026 | Fixed model-tab tool buttons (latent/observed) appearing dead on click in WebKit; blur probe now covers the canvas; update-check success now logged |
| v2.0.6 | 17 Sep 2026 | Effects/blur respects the OS reduce-transparency preference; fsQCA & .docx exports no longer fail silently; unbuildable-lock fix + CI consistency checks; Bandit job renaming |
| v2.0.5 | 5 Sep 2026 | Fixed the update indicator misreading GitHub 403s as offline; 6-hour update-check cache; distinct "rate limited" state |
| v2.0.4 | 4 Sep 2026 | Fixed a second launch killing the first one's server (single-instance lock, held port, shutdown race) |
| v2.0.3 | 5 Sep 2026 | Fixed tool buttons not repainting on WebKit plus slower launches; background-blur renderer probe with an Effects toggle |
| v2.0.2 | 5 Sep 2026 | Packaging toolchain separated from runtime deps; workflow_dispatch builds fixed; metadata/version consistency |
| v2.0.1 | 22 Aug 2026 | Fixed Bitcoin timestamping failing to initialize in packaged builds (OpenTimestamps) |
| v2.0.0 | 22 Aug 2026 | Bitcoin timestamping (OpenTimestamps) provenance across all analyses; method-name tooltips; fsQCA report/live-log fixes; dark-mode table fixes |
| v1.1.9 | 11 Aug 2026 | HOC syntax auto-detection with user guidance; Downloads tab renamed to Outputs |
| v1.1.5 LTS | 2 Aug 2026 | Report export rebuilds diagram from live theme before capture — fixes stale-theme exports |
| v1.1.4 | 2 Aug 2026 | Fixed HOC diagram auto-layout; hardcoded dark export background |
| v1.1.3 | 1 Aug 2026 | Formative (Mode B) PLS-SEM via `<~`; bootstrap SE/CI back-fill for HOC loadings |
| v1.1.2 | 1 Aug 2026 | HOC bootstrap significance back-fill; second-order diagram rendering fix |
| v1.1.1 | 7 Jul 2026 | Fixed pywebview startup and the Builder arrow tool |
| v1.1.0 | 5 Jul 2026 | Pinned mpmath to resolve a sympy dependency conflict |
| v1.0.0 LTS | 27 Jun 2026 | fsQCA (Quine-McCluskey), APA 7 Word export, schema freeze, 174 tests |
| v0.9.0 | 22 Jun 2026 | CVI, EFA, nomological validity, measurement invariance (MICOM), NCA-ESSE |
| v0.8.0 | 14 Jun 2026 | FIMIX-PLS, PLS-POS, PDF export |
| v0.7.0 | 8 Jun 2026 | Moderation, IPMA, NCA, conditional process (Hayes PROCESS 7/14/58/59) |
| v0.6.0 | 2 Jun 2026 | Higher-Order Constructs, MICOM, Multi-Group Analysis |
| v0.5.0 | 25 May 2026 | Predictive relevance (Q², PLSpredict, CVPAT), CMB marker analysis |
| v0.4.1 | 22 May 2026 | VIF, Cohen's f², indirect effects, outer weight significance |
| v0.3.0 | 11 May 2026 | AVE, CR, Cronbach's α, Fornell-Larcker, Validity tab |
| v0.2.0 | 7 May 2026 | Initial release — PLS/CB-SEM, HTMT, visual builder |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `server offline` in UI | Confirm `launcher.py` started cleanly; check port 8765 is free |
| `ModuleNotFoundError: semopy` | `uv sync --locked` from the project root |
| White screen (Linux) | Install WebKit2GTK — see [building.md](docs/building.md) |
| MSI build fails | Confirm WiX 3 on PATH: `candle.exe --version` |
| macOS "App is damaged" | `xattr -cr dist/NAVAL-SEM.app` |
| `.docx` export empty | Ensure a `/run` call completed for the current session before calling `/report` |

---

## Community

NAVAL-SEM is built with researchers and practitioners worldwide. Special thanks to everyone who reports bugs, tests releases, and suggests features.

→ [Meet contributors — Community Hall of Fame](https://github.com/navalsingh9/naval-sem/blob/master/CONTRIBUTING.md#community-recognition)
→ [Release roadmap](https://github.com/navalsingh9/naval-sem/blob/master/Product%20Roadmap.png?raw=true)
→ [Google Calendar](https://calendar.google.com/calendar/u/0?cid=YjZmYzkzMTBlYzQxZWQ5MDYxMDgwMDcyN2YwMjY0ZjliZDM1M2FiMjkzNjFlZjBlYjhmMGRkMWNhMmFiNWQ5MEBncm91cGNhci5nb29nbGUuY29t)

**Support the project:** [Donate via PayPal](https://www.paypal.com/paypalme/singhn9) · [Submit bug / feedback](https://forms.gle/N4AmCkJyCK6HHsZz8)

[![Liberapay receiving](https://img.shields.io/liberapay/receives/singhn9.svg?logo=liberapay)](https://liberapay.com/singhn9/donate)
---

## License

[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) — source-available, maintainer-controlled. Inspect, use, and share unmodified copies freely. Modified redistribution and commercial use beyond the licence terms are not permitted.

---

<sub>`PLS-SEM` · `fsQCA` · `structural equation modeling` · `SmartPLS alternative` · `free SEM software` · `offline SEM` · `CB-SEM` · `HTMT` · `mediation analysis` · `bootstrapping SEM` · `applied multivariate statistics`</sub>
