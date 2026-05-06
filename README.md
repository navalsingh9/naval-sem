# NAVAL-SEM

> Fully offline PLS-SEM / CB-SEM desktop application — visual model builder, bootstrapping, HTMT, and fit indices. No internet required after install.

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## Downloads

| Platform | File | Notes |
|---|---|---|
| Windows | `NAVAL-SEM-Setup.msi` | Recommended — installs with Start Menu + uninstall |
| Windows | `NAVAL-SEM.exe` | Portable — drop anywhere and run |
| macOS | `NAVAL-SEM.dmg` | Drag to Applications |
| Linux | `naval-sem_0.2.0_amd64.deb` | Debian/Ubuntu installer |
| Linux | `NAVAL-SEM` | Portable binary |

👉 **[Download from GitHub Releases](https://github.com/navalsingh9/naval-sem/releases)**

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

## Run in development

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
python launcher.py
# Opens http://127.0.0.1:8765 in a native window (pywebview)
# or in your default browser if pywebview is unavailable

# Dev mode — hot-reload backend, open browser manually:
uvicorn app.main:app --reload --port 8000
# Then open http://localhost:8000
```

---

## Package for release

### Windows — EXE + MSI

**One-time setup:**
```
# Install WiX 3 (MSI builder)
# Download from: https://github.com/wixtoolset/wix3/releases
# Install and add to PATH: C:\Program Files (x86)\WiX Toolset v3.14\bin
```

**Build:**
```
build_windows.bat
```

Outputs:
- `dist\NAVAL-SEM.exe` — portable single-file executable, no install needed
- `dist\NAVAL-SEM-Setup.msi` — proper Windows installer with Start Menu + uninstall

---

### macOS — .app + DMG

**One-time setup:**
```bash
brew install create-dmg
```

**Build:**
```bash
chmod +x build_macos.sh
./build_macos.sh
```

Outputs:
- `dist/NAVAL-SEM.app` — macOS application bundle
- `dist/NAVAL-SEM.dmg` — disk image for distribution

> **Gatekeeper note:** Without an Apple Developer ID, users must right-click → Open the first time. To properly code-sign, add your Developer ID to your Keychain before building.

---

### Linux — binary + .deb

**One-time setup:**
```bash
sudo apt install libgtk-3-0 libwebkit2gtk-4.0-dev \
  python3-gi python3-gi-cairo gir1.2-webkit2-4.0
```

**Build:**
```bash
chmod +x build_linux.sh
./build_linux.sh
```

Outputs:
- `dist/NAVAL-SEM` — portable ELF binary
- `dist/naval-sem_0.2.0_amd64.deb` — Debian/Ubuntu installer

Install the .deb:
```bash
sudo dpkg -i dist/naval-sem_0.2.0_amd64.deb
naval-sem
```

---

## Release to GitHub (automated)

The included GitHub Actions workflow (`.github/workflows/release.yml`) builds all three platforms automatically and publishes a GitHub Release.

**To trigger a release:**

```bash
# Bump the version in naval_sem.spec and the WiX .wxs file, then:
git tag v0.2.0
git push origin v0.2.0
```

GitHub Actions will:
1. Build EXE + MSI on Windows runner
2. Build .app + DMG on macOS runner
3. Build binary + .deb on Ubuntu runner
4. Create a GitHub Release and upload all 5 files automatically

---

## Reducing binary size (optional)

```bash
# Install UPX compressor — PyInstaller uses it automatically
# Windows: download from https://github.com/upx/upx/releases
# macOS:
brew install upx
# Linux:
sudo apt install upx
```

This typically reduces the EXE from ~120 MB to ~80 MB.

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

## Support Development

NAVAL-SEM is free to use. If you find it useful, please consider donating to help keep the project running and fund future features.

👉 [Donate via PayPal](https://www.paypal.com/paypalme/singhn9)

---

## Bug Reports & Feedback

Found a bug or have a suggestion? We read every submission!

👉 [Submit Bug Report / Feedback](https://forms.gle/N4AmCkJyCK6HHsZz8)

---

## Source Code Transparency

NAVAL-SEM is built from a public GitHub repository. All releases are generated automatically from that source — no closed build steps, no hidden binaries.

| Path | What it contains |
|---|---|
| `/app` | Backend engine — FastAPI routes, SEM fitting, data parsing, and response schemas |
| `/static/index.html` | Frontend UI — the full canvas builder, results panels, and all client-side logic |
| `/.github/workflows/release.yml` | Automated CI — builds all platforms (Windows, macOS, Linux) and publishes each GitHub Release |

You can audit, clone, or build from source at any time using the instructions in the sections above.

---

## License

This work is licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).

You are free to share this software with attribution for non-commercial purposes, without modifications.
