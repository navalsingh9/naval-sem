# Building NAVAL-SEM from Source

This document covers building the distributable installers locally. Most users should just [download a release](https://github.com/navalsingh9/naval-sem/releases/latest) instead.

---

## Prerequisites (all platforms)

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Windows — EXE + MSI

**One-time setup:**
1. Download [WiX Toolset v3](https://github.com/wixtoolset/wix3/releases)
2. Install and add to PATH: `C:\Program Files (x86)\WiX Toolset v3.14\bin`

**Build:**
```bat
build_windows.bat
```

Outputs:
- `dist\NAVAL-SEM.exe` — portable, no install needed
- `dist\NAVAL-SEM-Setup.msi` — Windows installer with Start Menu + uninstall

> The GitHub Actions CI runner builds these automatically on every tagged release. Local builds are only needed for testing.

---

## macOS — .app + DMG

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
- `dist/NAVAL-SEM.app`
- `dist/NAVAL-SEM.dmg`

> Without an Apple Developer ID, users must right-click → Open on first launch to bypass Gatekeeper.

---

## Linux — binary + .deb

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
- `dist/naval-sem_*.deb` — Debian/Ubuntu installer

Install the .deb:
```bash
sudo dpkg -i dist/naval-sem_*.deb
naval-sem
```

---

## Reducing binary size (optional)

```bash
# macOS / Linux:
brew install upx      # or: sudo apt install upx
# Windows: download from https://github.com/upx/upx/releases
```

PyInstaller uses UPX automatically if it's on PATH. Typically reduces EXE from ~120 MB to ~80 MB.

---

## Release process

Releases are fully automated via GitHub Actions (`.github/workflows/release.yml`).
All three platforms build in parallel on every version tag push.

```bash
# All feature work goes on v0.x-branch or dev, never directly to master

# Release day:
git checkout master
git merge v0.4-formative --no-ff -m "Release v0.4.0"
git tag v0.4.0
git push origin master
git push origin v0.4.0    # ← triggers CI build + GitHub Release
git checkout dev          # go back immediately
```

> **Rule:** `master` always equals the latest published release tag. Nothing else touches it.
