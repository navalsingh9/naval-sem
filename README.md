# NAVAL-SEM

> Fully offline PLS-SEM / CB-SEM desktop application — visual model builder, bootstrapping, HTMT, and fit indices. No internet required after install.

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## What is NAVAL-SEM?

NAVAL-SEM is a cross-platform desktop tool for Structural Equation Modelling (SEM). It provides a drag-and-drop canvas for building path models, runs PLS-SEM and CB-SEM (ML/WLS) estimation via [semopy](https://semopy.com/), and reports standardised path coefficients, fit indices (CFI, RMSEA, SRMR, χ²), bootstrapped confidence intervals, and HTMT discriminant-validity ratios — all without sending data to any server.

**Key capabilities:**

| Feature | Details |
|---|---|
| Model types | PLS-SEM, CB-SEM (ML), WLS |
| Data formats | CSV, Excel (.xlsx), SPSS (.sav) |
| Fit indices | CFI, RMSEA, SRMR, χ², df, p-value |
| Validity checks | HTMT ratios with configurable thresholds |
| Bootstrapping | Configurable resamples, 95 % CI on all paths |
| Model syntax | lavaan-compatible text syntax accepted as input |
| Deployment | Single-file executable — no Python or dependencies required |

---

## Screenshots

> *Screenshots and a demo GIF will be added here in an upcoming release.*

---

## Architecture overview

NAVAL-SEM uses a local client–server pattern entirely within the user's machine — no external network calls are ever made.

```
┌─────────────────────────────────────────────────┐
│                  NAVAL-SEM process               │
│                                                 │
│  ┌─────────────┐   HTTP (localhost:8765)         │
│  │  pywebview  │ ◄──────────────────────────┐   │
│  │  (native    │                            │   │
│  │   window)   │                            │   │
│  └─────────────┘                            │   │
│                                             │   │
│  ┌──────────────────────────────────────┐   │   │
│  │  FastAPI backend  (app/main.py)      │◄──┘   │
│  │  ├── /upload/preview  (parse file)   │       │
│  │  ├── /fit             (run SEM)      │       │
│  │  ├── /bootstrap       (resampling)   │       │
│  │  └── /htmt            (validity)     │       │
│  │                                      │       │
│  │  app/engine.py  ← semopy wrapper     │       │
│  │  app/parser.py  ← CSV/Excel/SPSS     │       │
│  └──────────────────────────────────────┘       │
│                                                 │
│  static/index.html  ← canvas UI (served above)  │
└─────────────────────────────────────────────────┘
```

`launcher.py` starts the FastAPI server as a background thread, then opens `static/index.html` inside a pywebview native window (or falls back to the default browser if pywebview is unavailable).

---

## Security & privacy model

- **Fully offline.** All computation runs on `localhost`. No data leaves the device.
- **No telemetry.** NAVAL-SEM does not collect usage statistics, crash reports, or any analytics.
- **No network permissions required.** The application binds only to `127.0.0.1:8765`.
- **Open source.** The complete source is available in this repository for independent audit.
- **Data handling.** Uploaded files are read into memory for the duration of the session and are never written to disk by the application.

---

## Source visibility

This repository contains the complete source code for NAVAL-SEM. You can inspect, audit, or build the application yourself from the files in this repository. See [Build from source](#build-from-source) below.

Contributions are not accepted under the current license (CC BY-NC-ND 4.0), but issues and feedback are welcome via the [feedback form](#bug-reports--feedback).

---

## Releases & downloads

Pre-built binaries for all platforms are published on the [GitHub Releases](https://github.com/navalsingh9/naval-sem/releases) page.

| Platform | File | Notes |
|---|---|---|
| Windows | `NAVAL-SEM-Setup.msi` | Recommended — installs with Start Menu + uninstall |
| Windows | `NAVAL-SEM.exe` | Portable — drop anywhere and run |
| macOS | `NAVAL-SEM.dmg` | Drag to Applications |
| Linux | `naval-sem_0.2.0_amd64.deb` | Debian/Ubuntu installer |
| Linux | `NAVAL-SEM` | Portable binary |

👉 **[Download from GitHub Releases](https://github.com/navalsingh9/naval-sem/releases)**

---

## Build from source

The build scripts use [PyInstaller](https://pyinstaller.org/) to produce self-contained binaries. All three platforms are also built automatically by GitHub Actions on every tagged release (see [`.github/workflows/release.yml`](.github/workflows/release.yml)).

### Prerequisites (all platforms)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Windows — EXE + MSI

**One-time setup:** Install [WiX Toolset v3](https://github.com/wixtoolset/wix3/releases) and add its `bin` directory to `PATH` (e.g. `C:\Program Files (x86)\WiX Toolset v3.14\bin`).

```bat
build_windows.bat
```

Outputs:
- `dist\NAVAL-SEM.exe` — portable single-file executable
- `dist\NAVAL-SEM-Setup.msi` — Windows installer with Start Menu entry + uninstaller

### macOS — .app + DMG

**One-time setup:**
```bash
brew install create-dmg
```

```bash
chmod +x build_macos.sh && ./build_macos.sh
```

Outputs:
- `dist/NAVAL-SEM.app` — macOS application bundle
- `dist/NAVAL-SEM.dmg` — disk image for distribution

> **Gatekeeper note:** Without an Apple Developer ID, users must right-click → Open the first time. To code sign properly, add your Developer ID to the Keychain before building.

### Linux — binary + .deb

**One-time setup:**
```bash
sudo apt install libgtk-3-0 libwebkit2gtk-4.0-dev \
  python3-gi python3-gi-cairo gir1.2-webkit2-4.0
```

```bash
chmod +x build_linux.sh && ./build_linux.sh
```

Outputs:
- `dist/NAVAL-SEM` — portable ELF binary
- `dist/naval-sem_0.2.0_amd64.deb` — Debian/Ubuntu installer

Install the `.deb`:
```bash
sudo dpkg -i dist/naval-sem_0.2.0_amd64.deb
naval-sem
```

### Reducing binary size (optional)

Installing [UPX](https://github.com/upx/upx/releases) before building lets PyInstaller compress the output automatically, typically reducing the EXE from ~120 MB to ~80 MB.

```bash
# macOS
brew install upx
# Linux
sudo apt install upx
# Windows — download the release zip and add to PATH
```

---

## Run in development

```bash
# Start the full app (native window via pywebview):
python launcher.py
# Opens http://127.0.0.1:8765 — falls back to browser if pywebview is unavailable.

# Hot-reload backend only (open browser manually):
uvicorn app.main:app --reload --port 8000
# Then open http://localhost:8000
```

---

## Project structure

```
naval-sem/
├── launcher.py              ← Entry point: starts server + opens UI window
├── requirements.txt         ← All Python dependencies
├── naval_sem.spec           ← PyInstaller build spec
├── build_windows.bat        ← Windows: builds EXE + MSI
├── build_macos.sh           ← macOS:   builds .app + DMG
├── build_linux.sh           ← Linux:   builds binary + .deb
│
├── app/
│   ├── main.py              ← FastAPI routes + static file serving
│   ├── engine.py            ← SEM fitting (PLS / CB / WLS), bootstrap, HTMT
│   ├── parser.py            ← CSV / Excel / SPSS file parser + lavaan syntax
│   └── schemas.py           ← Pydantic response models
│
├── static/
│   └── index.html           ← Full frontend (canvas builder, results panels)
│
├── installer/
│   ├── naval_sem.wxs        ← WiX MSI definition (Windows)
│   └── license.rtf          ← License shown in installer dialog
│
└── .github/
    └── workflows/
        └── release.yml      ← CI/CD: build all platforms + publish release
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `server offline` badge in UI | Make sure `launcher.py` started without errors; check port 8765 is free |
| `ModuleNotFoundError: semopy` | Run `pip install -r requirements.txt` in your venv |
| White screen / blank webview | pywebview needs WebKit2GTK on Linux — see Linux setup above |
| MSI build fails | Ensure WiX 3 is installed and on PATH: `candle.exe --version` should print a version |
| macOS "App is damaged" | Run `xattr -cr dist/NAVAL-SEM.app` to strip quarantine flags |
| Large binary size | Install UPX (see above) or add more entries to `excludes` in the spec |

---

## Support development

NAVAL-SEM is free to use. If you find it useful, please consider donating to help keep the project running and fund future features.

👉 [Donate via PayPal](https://www.paypal.com/paypalme/singhn9)

---

## Bug reports & feedback

Found a bug or have a suggestion? We read every submission.

👉 [Submit bug report / feedback](https://forms.gle/N4AmCkJyCK6HHsZz8)

---

## License

This work is licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).

You are free to share this software with attribution for non-commercial purposes, without modifications.
