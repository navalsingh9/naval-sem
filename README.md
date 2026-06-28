# NAVAL-SEM

> **Free offline PLS-SEM / CB-SEM / fsQCA desktop application** — visual model builder, bootstrapping, HTMT, APA 7th edition reporting, and R/Python export. No internet required. No licence. No sample size limit.

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Latest Release](https://img.shields.io/github/v/release/navalsingh9/naval-sem)](https://github.com/navalsingh9/naval-sem/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)](https://github.com/navalsingh9/naval-sem/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20124108.svg)](https://doi.org/10.5281/zenodo.20124108)
[![Discussions](https://img.shields.io/github/discussions/navalsingh9/naval-sem)](https://github.com/navalsingh9/naval-sem/discussions)

---

## Release Roadmap

![Roadmap](https://github.com/navalsingh9/naval-sem/blob/master/Product%20Roadmap.png?raw=true)

---

## Google Calendar

[Link](https://calendar.google.com/calendar/u/0?cid=YjZmYzkzMTBlYzQxZWQ5MDYxMDgwMDcyN2YwMjY0ZjliZDM1M2FiMjkzNjFlZjBlYjhmMGRkMWNhMmFiNWQ5MEBncm91cC5jYWxlbmRhci5nb29nbGUuY29t)

---

## Download

Get the latest NAVAL-SEM release for Windows, macOS, and Linux.

### Recommended Download

<a href="https://sourceforge.net/projects/naval-sem/files/latest/download" target="_blank">
  <img src="https://a.fsdn.com/con/app/sf-download-button" alt="Download NAVAL-SEM">
</a>

Fastest way to get the newest stable release.

### Release Mirrors

- SourceForge: https://sourceforge.net/projects/naval-sem/
- GitHub Releases: https://github.com/navalsingh9/naval-sem/releases/latest

| Platform | File | Notes |
|----------|------|-------|
| **Windows** | `NAVAL-SEM-Setup.msi` | Recommended — Start Menu integration + uninstall |
| **Windows** | `NAVAL-SEM.exe` | Portable — no installation required |
| **macOS** | `NAVAL-SEM.dmg` | Drag into Applications |
| **Linux** | `naval-sem_*.deb` | Debian / Ubuntu installer |
| **Linux** | `NAVAL-SEM` | Portable binary |

> **Windows:** SmartScreen may appear on first launch. Click *More info → Run anyway*.
> **macOS:** Right-click → *Open* on first launch to bypass Gatekeeper for unsigned apps.

---

## What is NAVAL-SEM?

NAVAL-SEM is a **structural equation modelling desktop app** that runs entirely on your machine. Load your dataset, draw your model in the visual canvas, and get bootstrapped path coefficients, HTMT, AVE, and fit indices — without an internet connection, a licence key, or your data leaving the machine.

**v1.0.0 LTS** — released 27 June 2026 — is the long-term support milestone. All public API schemas are frozen: no field removals or renames in patch or minor releases. This is the build to cite in your methods section.

Built for:
- **PhD students** who need full PLS-SEM without a SmartPLS subscription
- **Professors** who need a free, zero-install classroom tool
- **HR, Marketing, and Healthcare practitioners** running structural diagnostics on sensitive data
- **Anyone** who wants SmartPLS-quality output without the SmartPLS price

---

## Features

### Core SEM

- **PLS-SEM** — reflective + formative constructs, bootstrapped path coefficients (5,000 iterations), indirect effects, full/partial/no mediation classification
- **CB-SEM** — covariance-based SEM, lavaan syntax, fit indices (CFI, RMSEA, SRMR, χ²/df)
- **WLS** — weighted least squares estimator
- **Visual model builder** — drag-and-drop canvas, live validity warnings, Undo/Redo, PNG export
- **Measurement model** — AVE, Composite Reliability, Cronbach's α, outer loadings, cross-loadings, Fornell-Larcker criterion
- **HTMT** — full discriminant validity matrix, configurable threshold (0.85 / 0.90)

### Effects & Validity

- **VIF** — Variance Inflation Factor per indicator with strict (<3.3) and acceptable (<5.0) thresholds
- **Cohen's f²** — effect size per structural path (negligible / small / medium / large)
- **Mediation analysis** — bootstrapped specific indirect effects, 95% CI, mediation type classification
- **Predictive relevance** — Q² blindfolding, PLSpredict (k-fold), CVPAT (Liengaard et al. 2021)
- **Common Method Bias** — Lindell & Whitney (2001) marker variable analysis

### Advanced Analysis (v0.6–v0.8)

- **Higher-Order Constructs (HOC)** — repeated-indicator and two-stage methods
- **Multi-Group Analysis (MGA)** — compare path coefficients across groups with permutation significance testing
- **Moderation** — product-of-composites, simple slopes at ±1 SD, bootstrap CIs for interaction effects
- **Moderated Mediation** — conditional indirect effects, Index of Moderated Mediation, Hayes PROCESS Models 7, 14, 58/59
- **IPMA** — Importance–Performance Map Analysis for construct prioritisation
- **NCA** — Necessary Condition Analysis, CE-FDH and CR-FDH ceiling techniques with permutation significance
- **FIMIX-PLS** — EM-based finite mixture segmentation, AIC/BIC/CAIC model selection, segment membership
- **PLS-POS** — prediction-oriented segmentation
- **PDF export** — full results report to PDF

### Scale Development Suite (v0.9)

- **CVI** — Content Validity Index, item-level I-CVI and scale-level S-CVI/Ave and S-CVI/UA, Polit & Beck (2006) thresholds
- **EFA** — Exploratory Factor Analysis, principal-axis factoring, promax/varimax rotation, Kaiser criterion, scree eigenvalues
- **Nomological Validity** — bivariate correlation matrix, directional hypothesis testing with sign expectations
- **Measurement Invariance (MICOM extended)** — configural, compositional, and scalar invariance with permutation p-values
- **NCA-ESSE** — Effect Size Sensitivity Extension, threshold-removal sweep, joint-uniform benchmark (Becker, Richter, Ringle & Sarstedt, 2026)

### v1.0 LTS Additions

- **fsQCA** — fuzzy-set Qualitative Comparative Analysis: direct calibration, necessity analysis, truth table, Quine-McCluskey Boolean minimisation. Complex, parsimonious, and intermediate solutions with raw/unique coverage, consistency, and bubble-chart coincidence visualisation
- **APA 7th Edition Reporting** — one-click Word (.docx) export of measurement model, discriminant validity, structural paths, and indirect effects tables, formatted to journal submission standards
- **Schema freeze** — all public result models stable under semantic versioning; no field removals or renames in v1.x.x

### Export & Distribution

- **Code export** — R (lavaan / seminr syntax), Python (semopy syntax), JASP
- **Fully offline** — nothing leaves your machine, no account required, no internet after install
- **Citable** — Zenodo DOI archived for every release, `CITATION.cff` in repository root
- Available on **SourceForge**, **Microsoft Store**, and listed on **AlternativeTo**

---

## SEM Case Library

Five production-ready research cases — each with a real open dataset, construct specification, and expected path coefficients. Load any case to start a working model immediately.

| # | Case | Framework | Dataset | n | Key finding |
|---|------|-----------|---------|---|-------------|
| 01 | **Why Employees Really Quit** | JD-R Theory | IBM HR Attrition · Kaggle · Free | 1,470 | WLB→Intent β=0.08 n.s. once Manager Trust enters — full mediation |
| 02 | **Why Awareness Doesn't Become Purchase** | Aaker Brand Equity | Customer Personality Analysis · Kaggle · CC0 | 2,216 | Quality→Trust β=0.19 — the funnel gap regression misses |
| 03 | **Why Hospital Ratings Fall** | Modified SERVQUAL | HCAHPS · CMS.gov · U.S. Federal Public Domain | ~4,800 | Responsiveness→Overall β=0.11 n.s. — it's communication, not call speed |
| 04 | **Why Fintech Adoption Stalls** | UTAUT + Trust Extension | World Bank Findex 2021 · CC BY 4.0 | 3,212 | Performance Expectancy direct β=0.17 — Institutional Trust is the real barrier |
| 05 | **Why Students Drop MOOCs** | TAM + Self-Determination Theory | OULAD · Open University UK · CC BY 4.0 | 32,593 | Social Belonging β=0.48 — stronger than Perceived Usefulness |

Each case ships with a prepared CSV, indicator mapping, and the "surprise finding" — the path that collapses under mediation and changes the practical recommendation.

→ **[Browse `cases/` →](https://github.com/navalsingh9/naval-sem/tree/master/cases)**

---

## How it works

NAVAL-SEM starts a local FastAPI server on `http://127.0.0.1:8765` and opens the interface in a native window via pywebview. All computation — engine, data, results — stays on your machine.

```
launcher.py
  ├── starts FastAPI on port 8765
  ├── opens pywebview window  →  http://127.0.0.1:8765
  ├── app/engine.py           ← PLS/CB-SEM, bootstrapping, HTMT, MGA
  ├── app/fsqca.py            ← fsQCA engine (v1.0)
  ├── app/report.py           ← APA 7th edition DOCX export (v1.0)
  ├── app/parser.py           ← CSV / Excel / SPSS ingestion
  └── static/index.html       ← canvas builder, results panels, export
```

No telemetry. No account. No data transmission.

---

## Run from source

```bash
git clone https://github.com/navalsingh9/naval-sem.git
cd naval-sem

uv sync                         # creates .venv and installs locked dependencies
uv run launcher.py              # opens at http://127.0.0.1:8765
```

No `uv`? [Install it](https://docs.astral.sh/uv/getting-started/installation/) first — it's what keeps every contributor's environment identical via `uv.lock`.

→ For full build instructions (EXE, DMG, .deb): see [`docs/building.md`](docs/building.md)

---

## NAVAL-SEM vs SmartPLS

| | NAVAL-SEM v1.0 LTS | SmartPLS 4 |
|--|-----------|------------|
| Price | **Free** | Paid licence |
| Sample size limit | **None** | Student edition: 100 rows |
| Construct limit | **None** | Student edition: 4 constructs |
| OS | **Windows · macOS · Linux** | Windows · macOS only |
| Offline | **Fully offline — no internet ever** | Offline after licence activation |
| PLS-SEM | **✓ Full support** | ✓ Full support |
| CB-SEM | **✓ ML, WLS estimators** | Limited |
| Moderation & IPMA | **✓ Full support** | ✓ Full support |
| NCA | **✓ CE-FDH, CR-FDH, NCA-ESSE** | ✓ Basic support |
| HOC | **✓ Repeated indicator + two-stage** | ✓ Full support |
| MGA | **✓ Permutation significance** | ✓ Full support |
| FIMIX-PLS / PLS-POS | **✓ Full support** | ✓ Full support |
| Scale development (CVI, EFA, Nomological, MICOM) | **✓ Full suite (v0.9)** | Partial |
| **fsQCA** | **✓ Quine-McCluskey minimisation (v1.0)** | Not available |
| **APA 7 reporting (.docx)** | **✓ One-click Word export (v1.0)** | Not available |
| R / Python export | **✓ lavaan, seminr, semopy** | Not available |
| Case library | **✓ 5 cases, open datasets, expected findings** | Sample projects only |
| Data privacy | **Localhost only — nothing transmitted** | Local analysis |
| Schema stability | **✓ Frozen public API (v1.0 LTS)** | Proprietary |
| Citable DOI | **✓ Zenodo archive per release** | Not available |

---

## Validation

NAVAL-SEM v1.0 ships with **174 pytest + Playwright tests** gated on every release. Results are compared to published anchor values from peer-reviewed literature — if the numbers don't match, the release is blocked.

Key benchmarks:
- HS1939 CB-SEM CFI ≈ 0.931 (Holzinger & Swineford 1939)
- Bollen Political Democracy CFI ≥ 0.997 (Bollen 1989)
- Corporate Reputation avg loading ≈ 0.80, max HTMT ≈ 0.86 (Hair et al. 2011/2013)
- fsQCA consistency ≥ 0.80 (Wagemann & Schneider 2010; Ragin 2008)

→ [View full test suite documentation](https://naval-sem.sourceforge.io/testbench.html)

---

## Citation

If you use NAVAL-SEM in published research, please cite:

```
Singh, N. (2026). NAVAL-SEM: Free offline structural equation modelling
desktop application [Software, v1.0.0 LTS].
https://doi.org/10.5281/zenodo.20124108
```

```bibtex
@software{singh2026navalsem,
  author  = {Singh, Naval},
  title   = {{NAVAL-SEM}: Free offline structural equation modelling desktop application},
  year    = {2026},
  doi     = {10.5281/zenodo.20124108},
  url     = {https://github.com/navalsingh9/naval-sem},
  version = {1.0.0},
  license = {CC BY-NC-ND 4.0}
}
```

A `CITATION.cff` is included in the repository root for APA 7, BibTeX, and RIS export.

---

## Changelog highlights

| Version | Date | Additions |
|---------|------|-----------|
| **v1.0.0 LTS** | 27 Jun 2026 | fsQCA (Quine-McCluskey), APA 7 Word export, schema freeze, 174 tests |
| v0.9.0 | 22 Jun 2026 | CVI, EFA, Nomological validity, Measurement invariance (MICOM), NCA-ESSE |
| v0.8.0 | 14 Jun 2026 | FIMIX-PLS, PLS-POS, PDF export |
| v0.7.0 | 8 Jun 2026 | Moderation, IPMA, NCA, Conditional process (Hayes PROCESS Models 7/14/58/59) |
| v0.6.0 | 2 Jun 2026 | Higher-Order Constructs, MICOM, Multi-Group Analysis |
| v0.5.0 | 25 May 2026 | Predictive relevance (Q², PLSpredict, CVPAT), CMB marker analysis |
| v0.4.1 | 22 May 2026 | VIF, Cohen's f², indirect effects, outer weight significance |
| v0.3.0 | 11 May 2026 | AVE, CR, Cronbach's α, Fornell-Larcker, Validity tab |
| v0.2.0 | 7 May 2026 | Initial release — PLS/CB-SEM, HTMT, visual builder |

→ [Full CHANGELOG](CHANGELOG.md)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `server offline` in UI | Confirm `launcher.py` started cleanly; check port 8765 is free |
| `ModuleNotFoundError: semopy` | `uv sync --locked` from the project root |
| White screen (Linux) | Install WebKit2GTK — see [building.md](docs/building.md) |
| MSI build fails | Confirm WiX 3 on PATH: `candle.exe --version` |
| macOS "App is damaged" | `xattr -cr dist/NAVAL-SEM.app` |
| DOCX export empty | Ensure a `/run` call completed for the current session before calling `/report` |

---

## Support

👉 [Donate via PayPal](https://www.paypal.com/paypalme/singhn9) — helps keep NAVAL-SEM free
👉 [Submit bug / feedback](https://forms.gle/N4AmCkJyCK6HHsZz8)

---

## License

[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) — NAVAL-SEM is source-available, maintainer-controlled software. Users may inspect, use, and share unmodified copies under the CC BY-NC-ND 4.0 license. Modified redistributions and commercial use are not permitted. Bug reports and pull requests are welcome, but only official releases published by the maintainer are authorized NAVAL-SEM distributions.

---

## Keywords

`PLS-SEM` · `fsQCA` · `structural equation modeling` · `SmartPLS alternative` · `SmartPLS free` · `free SEM software` · `offline SEM` · `CB-SEM` · `HTMT` · `mediation analysis` · `bootstrapping SEM` · `APA reporting` · `SEM desktop app` · `lavaan` · `semopy` · `CVI` · `EFA` · `measurement invariance` · `NCA` · `FIMIX-PLS` · `fuzzy-set QCA` · `HR analytics SEM` · `brand equity SEM` · `UTAUT PLS-SEM` · `TAM SEM` · `SERVQUAL PLS-SEM` · `structural equation modeling Python` · `PLS-SEM Windows macOS Linux`
