#!/usr/bin/env bash
# ============================================================
#  NAVAL-SEM — Linux Build Script
#  Produces:
#    dist/NAVAL-SEM           (portable ELF binary)
#    dist/naval-sem_0.4.0_amd64.deb   (Debian/Ubuntu package)
#
#  Prerequisites:
#    pip install pyinstaller pywebview
#    sudo apt install python3-gi python3-gi-cairo gir1.2-webkit2-4.0
#      (needed by pywebview on Linux/GTK)
# ============================================================

set -e
cd "$(dirname "$0")"

VERSION="0.4.0"
ARCH="amd64"
PKG_NAME="naval-sem"
DEB_DIR="dist/deb_pkg"

echo ""
echo " ========================================="
echo "  NAVAL-SEM Linux Build"
echo " ========================================="
echo ""

# ── 1. Dependencies ───────────────────────────────────────────────────────────
echo " [1/4] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# ── 2. PyInstaller binary ─────────────────────────────────────────────────────
echo " [2/4] Building binary with PyInstaller..."
pyinstaller naval_sem.spec --clean --noconfirm
echo " Binary → dist/NAVAL-SEM"

# ── 3. .deb package ──────────────────────────────────────────────────────────
echo " [3/4] Assembling .deb package..."

# Directory structure
rm -rf "$DEB_DIR"
mkdir -p "$DEB_DIR/DEBIAN"
mkdir -p "$DEB_DIR/usr/bin"
mkdir -p "$DEB_DIR/usr/share/applications"
mkdir -p "$DEB_DIR/usr/share/icons/hicolor/256x256/apps"

# Copy binary
cp dist/NAVAL-SEM "$DEB_DIR/usr/bin/naval-sem"
chmod +x "$DEB_DIR/usr/bin/naval-sem"

# .desktop file (shows in app launcher)
cat > "$DEB_DIR/usr/share/applications/naval-sem.desktop" << 'DESKTOP'
[Desktop Entry]
Type=Application
Name=NAVAL-SEM
Comment=Structural Equation Modelling Desktop App
Exec=/usr/bin/naval-sem
Icon=naval-sem
Terminal=false
Categories=Science;Education;
Keywords=SEM;PLS;statistics;research;
DESKTOP

# Copy icon if present
if [ -f "naval_sem_256.png" ]; then
  cp naval_sem_256.png "$DEB_DIR/usr/share/icons/hicolor/256x256/apps/naval-sem.png"
fi

# DEBIAN/control
cat > "$DEB_DIR/DEBIAN/control" << CTRL
Package: $PKG_NAME
Version: $VERSION
Architecture: $ARCH
Maintainer: NAVAL-SEM Project <contact@naval-sem.io>
Description: NAVAL-SEM — Structural Equation Modelling Desktop App
 Open-source PLS-SEM and CB-SEM desktop application with
 visual model builder, bootstrapping, HTMT analysis, and
 support for CSV, Excel, and SPSS data files.
 All computation runs fully offline.
Depends: libgtk-3-0, libwebkit2gtk-4.0-37
Homepage: https://github.com/your-org/naval-sem
Section: science
Priority: optional
CTRL

# Build .deb
dpkg-deb --build "$DEB_DIR" "dist/${PKG_NAME}_${VERSION}_${ARCH}.deb"
echo " .deb  → dist/${PKG_NAME}_${VERSION}_${ARCH}.deb"

# ── 4. AppImage (optional) ────────────────────────────────────────────────────
echo " [4/4] Checking for AppImage tools..."
if command -v appimagetool &>/dev/null; then
  mkdir -p dist/AppDir/usr/bin dist/AppDir/usr/share/applications
  cp dist/NAVAL-SEM dist/AppDir/usr/bin/naval-sem
  cp "$DEB_DIR/usr/share/applications/naval-sem.desktop" dist/AppDir/
  [ -f "naval_sem_256.png" ] && cp naval_sem_256.png dist/AppDir/naval-sem.png
  appimagetool dist/AppDir "dist/NAVAL-SEM-${VERSION}-x86_64.AppImage"
  echo " AppImage → dist/NAVAL-SEM-${VERSION}-x86_64.AppImage"
else
  echo " appimagetool not found — skipping AppImage."
  echo " Download from https://github.com/AppImage/AppImageKit/releases"
fi

echo ""
echo " Done!"
echo "  dist/NAVAL-SEM                 (portable binary)"
echo "  dist/${PKG_NAME}_${VERSION}_${ARCH}.deb"
echo ""
