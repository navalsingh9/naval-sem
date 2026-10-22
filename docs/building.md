# Building NAVAL-SEM from Source

This document covers the supported local packaging workflow for the project. Most users should just [download a release](https://github.com/navalsingh9/naval-sem/releases/latest) instead of building from source.

---

## Prerequisites

Use the repository's locked environment and the build dependency group so local builds match the release pipeline:

```bash
uv sync --locked --group build
```

This installs the runtime dependencies plus the PyInstaller toolchain used by the release jobs in [release.yml](../.github/workflows/release.yml).

---

## Windows — EXE + MSI

This repo does not include a checked-in Windows packaging script. The official Windows workflow is the same one used by the CI pipeline in [release.yml](../.github/workflows/release.yml): build the PyInstaller EXE, then assemble the MSI with WiX.

**One-time setup:**
1. Install [WiX Toolset v3](https://github.com/wixtoolset/wix3/releases)
2. Add the WiX `bin` folder to `PATH` (for example: `C:\Program Files (x86)\WiX Toolset v3.14\bin`)

**Build:**
```powershell
uv sync --locked --group build
uv run pyinstaller naval_sem.spec --clean --noconfirm
$wixPath = Get-ChildItem "C:\Program Files (x86)" -Directory | Where-Object { $_.Name -like "WiX*" } | Select-Object -First 1 -ExpandProperty FullName
$env:Path += ";$wixPath\bin"

candle.exe installer\naval_sem.wxs -o installer\naval_sem.wixobj "-dProductVersion=2.0.11"
light.exe installer\naval_sem.wixobj -o dist\NAVAL-SEM-Setup.msi -ext WixUIExtension
```

Outputs:
- `dist\NAVAL-SEM.exe` — portable, no install required
- `dist\NAVAL-SEM-Setup.msi` — installer with Start Menu + uninstall entry

> The GitHub Actions release job builds these automatically on every version tag. Local builds are mainly for testing or debugging packaging issues.

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

## Release process

The release pipeline is triggered by version tags such as `v2.0.11` and then builds Windows, macOS, and Linux artifacts in parallel via `.github/workflows/release.yml`.

```bash
git checkout main
# make the release changes and update the version as needed
git tag v2.0.11
git push origin main
git push origin v2.0.11
```

The CI workflow injects the tag into the app version, runs PyInstaller, and uploads the packaged artifacts to the GitHub Release.

---

## Reducing binary size (optional)

```bash
# macOS / Linux:
brew install upx      # or: sudo apt install upx
# Windows: download from https://github.com/upx/upx/releases
```

PyInstaller uses UPX automatically if it is on `PATH`. It typically reduces the EXE from roughly 120 MB to around 80 MB.
