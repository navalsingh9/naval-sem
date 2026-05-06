# NAVAL-SEM

> Fully offline PLS-SEM / CB-SEM desktop application — visual model builder, bootstrapping, HTMT, and fit indices. No internet required after install.

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
│   └── naval_sem.wxs        ← WiX MSI definition (Windows)
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
# Install WiX 4 (MSI builder)
dotnet tool install --global wix
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
| MSI build fails | Ensure WiX 4 is installed: `wix --version` should print a version |
| macOS "App is damaged" | Run `xattr -cr dist/NAVAL-SEM.app` to strip quarantine flags |
| Large binary size | Install UPX (see above) or add more entries to `excludes` in the spec |
