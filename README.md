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

## What's new in v2.0.0

**Bitcoin timestamping (OpenTimestamps) provenance** — optional, off by default. Turn it on and every analysis gets a cryptographic fingerprint anchored to the Bitcoin blockchain via the free [OpenTimestamps](https://opentimestamps.org/) calendar service, embedded in every exported artifact (CSV, JSON, R/Python/lavaan code, APA `.docx`, PDF report). No cryptocurrency is bought, held, traded, or mined — this only proves *when* a result existed, nothing more. See [Provenance & integrity](#provenance--integrity) below.

Also in this release:
- Hover tooltips spelling out every abbreviated method (MGA, HOC, IPMA, NCA, FIMIX, PLS-POS, NCA-ESSE, EFA, CVI, fsQCA, HTMT)
- fsQCA: missing Report-tab entry, live-log wiring, and full report section (necessity table + all three solutions) fixed
- Consistent necessity-label colors between Analyse and Report tabs
- Dark-mode table borders and right-aligned numeric headers

<sub>Full history in [Changelog](#changelog).</sub>

---

## Why NAVAL-SEM

Built for people who need real SEM output without a subscription or a data-privacy trade-off:

- **PhD students** who need full PLS-SEM without a SmartPLS licence
- **Professors** who want a free, zero-install classroom tool
- **HR, marketing, and healthcare practitioners** running structural diagnostics on sensitive data that can't leave the machine
- **Anyone** who wants SmartPLS-quality output without the SmartPLS price

NAVAL-SEM starts a local FastAPI server on `127.0.0.1:8765` and opens the UI in a native window. Nothing is transmitted anywhere unless you explicitly opt into timestamping a fingerprint hash — your raw data never does.

**v2.0.0** is the current stable release. **v1.1.5 LTS** (2 Aug 2026) remains the long-term-support milestone with frozen public API schemas — cite this build in your methods section if schema stability matters more than new features.

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
| Sample size / construct limit | **None** | Student edition: 100 rows / 4 constructs |
| OS | **Windows · macOS · Linux** | Windows · macOS only |
| Offline | **Always** | After licence activation |
| CB-SEM | **ML + WLS estimators** | Limited |
| fsQCA | **Quine-McCluskey minimisation** | Not available |
| APA 7 reporting | **One-click `.docx`** | Not available |
| R / Python export | **lavaan, seminr, semopy** | Not available |
| Result provenance | **Optional Bitcoin timestamping** | Not available |
| Data privacy | **Localhost only** | Local analysis |
| Schema stability | **Frozen public API (LTS)** | Proprietary |
| Citable DOI | **Zenodo, every release** | Not available |

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

Every result NAVAL-SEM produces can optionally carry a **fingerprint** (a hash of the analysis inputs and outputs) that gets timestamped against the Bitcoin blockchain via the free, public [OpenTimestamps](https://opentimestamps.org/) calendar network. This is entirely opt-in and off by default.

What it does and doesn't mean:
- **Does:** prove a specific result existed at or before a specific point in time, independently verifiable by anyone, forever.
- **Doesn't:** involve buying, holding, trading, or mining any cryptocurrency. No wallet, no funds, no account.
- **Doesn't:** transmit your data — only a hash (fingerprint) leaves your machine, never the underlying dataset.

The fingerprint and anchor status are embedded in every export format: CSV, JSON, R/Python/lavaan code, the APA `.docx` report, and the PDF report. Standalone reports (e.g. a fsQCA-only PDF) carry their own correctly scoped fingerprint rather than inheriting the main run's.

```
launcher.py
  ├── starts FastAPI on port 8765
  ├── opens pywebview window   →  http://127.0.0.1:8765
  ├── app/engine.py            ← PLS/CB-SEM, bootstrapping, HTMT, MGA
  ├── app/fsqca.py             ← fsQCA engine
  ├── app/anchor.py            ← optional fingerprint + OpenTimestamps anchoring
  ├── app/report.py            ← APA 7th-edition DOCX export
  ├── app/parser.py            ← CSV / Excel / SPSS ingestion
  └── static/index.html        ← canvas builder, results panels, export
```

No telemetry. No account. Anchoring sends a hash and nothing else — see [What touches the network](#what-touches-the-network) for the complete list of outbound requests the application can make.

---

## Run from source

```bash
git clone https://github.com/navalsingh9/naval-sem.git
cd naval-sem

uv sync                # creates .venv, installs locked dependencies
uv run launcher.py     # opens at http://127.0.0.1:8765
```

No `uv`? [Install it](https://docs.astral.sh/uv/getting-started/installation/) first — it's what keeps every contributor's environment identical via `uv.lock`.

→ Full build instructions (EXE, DMG, `.deb`): [`docs/building.md`](docs/building.md)

---

## Validation

Every release is gated on **174 pytest + Playwright tests**, compared against published anchor values from peer-reviewed literature. If the numbers don't match, the release is blocked.

- HS1939 CB-SEM CFI ≈ 0.931 (Holzinger & Swineford 1939)
- Bollen Political Democracy CFI ≥ 0.997 (Bollen 1989)
- Corporate Reputation avg loading ≈ 0.80, max HTMT ≈ 0.86 (Hair et al. 2011/2013)
- fsQCA consistency ≥ 0.80 (Wagemann & Schneider 2010; Ragin 2008)

Beyond those anchors, the suite checks FIML log-likelihood against Arbuckle (1996), fit indices against the `lavaan` tutorial's worked examples, Mardia's test on known normal and non-normal data, and ground-truth recovery for the Bayesian, latent-class, multi-group and PLSpredict paths — that is, whether the estimator finds a structure that was deliberately planted, and correctly finds nothing in noise.

→ [Full test suite documentation](https://naval-sem.sourceforge.io/testbench.html)

**Requesting the suite.** The test code itself is not published, to limit derivative redistribution under the licence below. If you are evaluating NAVAL-SEM for research use and want to run the tests yourself, email **navalsem@hotmail.com** from an institutional or otherwise verifiable address and ask for a copy. Requests from anonymous or disposable addresses are not answered.

---

## What touches the network

NAVAL-SEM performs **all analysis offline**. No dataset, variable name, model, or result ever leaves your machine, and every statistical feature works with networking disabled. Three optional features do make outbound requests, listed here in full so the answer is on record for ethics and IRB review:

| Feature | When | Where it connects | What is sent |
|---|---|---|---|
| Update check | ~4 s after launch, and on demand via the toolbar button | `api.github.com` | Nothing but the request itself. Compares the latest release tag to your version. |
| Result anchoring | Only when you tick **anchor** on a run — off by default | OpenTimestamps calendars: `a.pool.opentimestamps.org`, `b.pool.opentimestamps.org`, `a.pool.eternitywall.com`, `ots.btc.catallaxy.com` | A SHA-256 fingerprint only. Never the dataset, the model, or the results. No wallet, no funds, no account. |
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

---

## Changelog

| Version | Date | Highlights |
|---|---|---|
| **v2.0.0** | 22 Aug 2026 | Bitcoin timestamping (OpenTimestamps) provenance across all analyses; method-name tooltips; fsQCA report/live-log fixes; dark-mode table fixes |
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
→ [Google Calendar](https://calendar.google.com/calendar/u/0?cid=YjZmYzkzMTBlYzQxZWQ5MDYxMDgwMDcyN2YwMjY0ZjliZDM1M2FiMjkzNjFlZjBlYjhmMGRkMWNhMmFiNWQ5MEBncm91cC5jYWxlbmRhci5nb29nbGUuY29t)

**Support the project:** [Donate via PayPal](https://www.paypal.com/paypalme/singhn9) · [Submit bug / feedback](https://forms.gle/N4AmCkJyCK6HHsZz8)

---

## License

[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) — source-available, maintainer-controlled. Inspect, use, and share unmodified copies freely. Modified redistribution and commercial use are not permitted. Bug reports and pull requests are welcome, but only official releases published by the maintainer are authorized NAVAL-SEM distributions.

---

<sub>`PLS-SEM` · `fsQCA` · `structural equation modeling` · `SmartPLS alternative` · `free SEM software` · `offline SEM` · `CB-SEM` · `HTMT` · `mediation analysis` · `bootstrapping SEM` · `APA reporting` · `lavaan` · `semopy` · `CVI` · `EFA` · `measurement invariance` · `NCA` · `FIMIX-PLS` · `fuzzy-set QCA` · `UTAUT PLS-SEM` · `TAM SEM` · `SERVQUAL PLS-SEM`</sub>
