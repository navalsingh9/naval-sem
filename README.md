# NAVAL-SEM

> **Free offline PLS-SEM / CB-SEM desktop application** — visual model builder, bootstrapping, HTMT, fit indices, and R/Python export. No internet required. No licence. No sample size limit.

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Latest Release](https://img.shields.io/github/v/release/navalsingh9/naval-sem)](https://github.com/navalsingh9/naval-sem/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)](https://github.com/navalsingh9/naval-sem/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20124108.svg)](https://doi.org/10.5281/zenodo.20124108)
[![Discussions](https://img.shields.io/github/discussions/navalsingh9/naval-sem)](https://github.com/navalsingh9/naval-sem/discussions)
[![Download NAVAL-SEM](https://a.fsdn.com/con/app/sf-download-button)](https://sourceforge.net/projects/naval-sem/)
---

## Google Calendar: 
[Link](https://calendar.google.com/calendar/u/0?cid=YjZmYzkzMTBlYzQxZWQ5MDYxMDgwMDcyN2YwMjY0ZjliZDM1M2FiMjkzNjFlZjBlYjhmMGRkMWNhMmFiNWQ5MEBncm91cC5jYWxlbmRhci5nb29nbGUuY29t)

## Download

| Platform | File | Notes |
|----------|------|-------|
| **Windows** | `NAVAL-SEM-Setup.msi` | Recommended — Start Menu + uninstall |
| **Windows** | `NAVAL-SEM.exe` | Portable — run anywhere, no install needed |
| **macOS** | `NAVAL-SEM.dmg` | Drag to Applications |
| **Linux** | `naval-sem_*.deb` | Debian/Ubuntu installer |
| **Linux** | `NAVAL-SEM` | Portable binary |

👉 **[Download latest release →](https://github.com/navalsingh9/naval-sem/releases/latest)**

> **Windows:** SmartScreen may warn on first run — click *More info → Run anyway*. Normal for unsigned apps.
> **macOS:** Right-click → Open the first time to bypass Gatekeeper.

---

## What is NAVAL-SEM?

NAVAL-SEM is a **structural equation modelling desktop app** that runs entirely on your machine. Load your dataset, draw your model in the visual canvas, and get bootstrapped path coefficients, HTMT, AVE, and fit indices — without an internet connection, a licence key, or your data leaving the machine.

Built for:
- **PhD students** who need full PLS-SEM without a SmartPLS subscription
- **Professors** who need a free, zero-install classroom tool
- **HR, Marketing, and Healthcare practitioners** running structural diagnostics on sensitive data
- **Anyone** who wants SmartPLS-quality output without the SmartPLS price

---

## Features

- **PLS-SEM** — reflective + formative constructs, bootstrapped path coefficients (5,000 iterations), indirect effects, full/partial/no mediation classification
- **CB-SEM** — covariance-based SEM, lavaan syntax, fit indices (CFI, RMSEA, SRMR, χ²/df)
- **Visual model builder** — drag-and-drop canvas, live validity warnings
- **Measurement model** — AVE, Composite Reliability, Cronbach's α, outer loadings, cross-loadings
- **HTMT** — full discriminant validity matrix, configurable threshold (0.85 / 0.90)
- **Mediation analysis** — bootstrapped specific indirect effects, 95% CI, mediation type classification
- **Multi-group analysis (MGA)** — compare path coefficients across groups
- **Export** — R (lavaan / seminr syntax), Python (semopy syntax), CSV tables, PDF report
- **Fully offline** — nothing leaves your machine, no account required, no internet after install

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
  ├── app/engine.py    ← PLS/CB-SEM, bootstrapping, HTMT, MGA
  ├── app/parser.py    ← CSV / Excel / SPSS ingestion
  └── static/index.html  ← canvas builder, results panels, export
```

No telemetry. No account. No data transmission.

---

## Run from source

```bash
git clone https://github.com/navalsingh9/naval-sem.git
cd naval-sem

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python launcher.py              # opens at http://127.0.0.1:8765
```

→ For full build instructions (EXE, DMG, .deb): see [`docs/building.md`](docs/building.md)

---

## NAVAL-SEM vs SmartPLS

| | NAVAL-SEM | SmartPLS 4 |
|--|-----------|------------|
| Price | **Free** | Paid licence |
| Sample size limit | **None** | Student edition: 100 rows |
| Construct limit | **None** | Student edition: 4 constructs |
| OS | **Windows · macOS · Linux** | Windows · macOS only |
| Offline | **Fully offline — no internet ever** | Offline after licence activation |
| Case library | **5 cases, open datasets, expected findings** | Sample projects only |
| R / Python export | **✓ lavaan, seminr, semopy** | Not available |
| Data privacy | **Localhost only — nothing transmitted** | Local analysis |
| Citation count | Growing (2026 launch) | ~50,000+ (established) |
| Moderation support | In development | Full support |

---

## Citation

```
Singh, N. (2025). NAVAL-SEM: Free offline structural equation modelling
desktop application [Software, v0.4.0].
https://doi.org/10.5281/zenodo.20124109
```

```bibtex
@software{singh2025navalsem,
  author  = {Singh, Naval},
  title   = {{NAVAL-SEM}: Free offline structural equation modelling desktop application},
  year    = {2025},
  doi     = {10.5281/zenodo.20124109},
  url     = {https://github.com/navalsingh9/naval-sem},
  version = {0.4.0},
  license = {CC BY-NC-ND 4.0}
}
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `server offline` in UI | Confirm `launcher.py` started cleanly; check port 8765 is free |
| `ModuleNotFoundError: semopy` | `pip install -r requirements.txt` inside your venv |
| White screen (Linux) | Install WebKit2GTK — see [building.md](docs/building.md) |
| MSI build fails | Confirm WiX 3 on PATH: `candle.exe --version` |
| macOS "App is damaged" | `xattr -cr dist/NAVAL-SEM.app` |

---

## Support

👉 [Donate via PayPal](https://www.paypal.com/paypalme/singhn9) — helps keep NAVAL-SEM free
👉 [Submit bug / feedback](https://forms.gle/N4AmCkJyCK6HHsZz8)

---

## License

[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) — free to use and share with attribution, non-commercial, no modifications.

---

## Keywords

`PLS-SEM` · `structural equation modeling` · `SmartPLS alternative` · `SmartPLS free` · `free SEM software` · `offline SEM` · `CB-SEM` · `HTMT` · `mediation analysis` · `bootstrapping SEM` · `SEM desktop app` · `lavaan` · `semopy` · `HR analytics SEM` · `brand equity SEM` · `UTAUT PLS-SEM` · `TAM SEM` · `SERVQUAL PLS-SEM` · `structural equation modeling Python` · `PLS-SEM Windows macOS Linux`
