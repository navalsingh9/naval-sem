#!/usr/bin/env bash
# ============================================================
#  NAVAL-SEM — macOS Build Script
#  Produces:
#    dist/NAVAL-SEM.app     (macOS application bundle)
#    dist/NAVAL-SEM.dmg     (disk image for distribution)
#
#  Prerequisites:
#    pip install pyinstaller pywebview
#    brew install create-dmg   (for DMG packaging)
# ============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo " ========================================="
echo "  NAVAL-SEM macOS Build"
echo " ========================================="
echo ""

# ── 1. Dependencies ───────────────────────────────────────────────────────────
echo " [1/4] Installing dependencies..."
pip install -r requirements.txt --quiet

# ── 2. Build .app with PyInstaller ────────────────────────────────────────────
echo " [2/4] Building .app bundle..."
pyinstaller naval_sem.spec \
  --clean --noconfirm \
  --windowed \
  --name "NAVAL-SEM" \
  --osx-bundle-identifier "io.naval-sem.app"

echo " App → dist/NAVAL-SEM.app"

# ── 3. Code-sign (optional — skip if no developer cert) ──────────────────────
if security find-identity -v -p codesigning | grep -q "Developer ID"; then
  echo " [3/4] Code-signing..."
  IDENTITY=$(security find-identity -v -p codesigning | grep "Developer ID" | head -1 | awk '{print $2}')
  codesign --deep --force --verify --verbose \
    --sign "$IDENTITY" \
    --options runtime \
    dist/NAVAL-SEM.app
  echo " Signed with: $IDENTITY"
else
  echo " [3/4] No Developer ID cert found — skipping code-sign."
  echo "       Users will need to right-click → Open to bypass Gatekeeper."
fi

# ── 4. Package as DMG ────────────────────────────────────────────────────────
echo " [4/4] Creating DMG..."
if command -v create-dmg &>/dev/null; then
  create-dmg \
    --volname "NAVAL-SEM" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "NAVAL-SEM.app" 175 190 \
    --hide-extension "NAVAL-SEM.app" \
    --app-drop-link 425 190 \
    "dist/NAVAL-SEM.dmg" \
    "dist/NAVAL-SEM.app"
  echo " DMG  → dist/NAVAL-SEM.dmg"
else
  # Fallback: plain hdiutil DMG
  hdiutil create -volname "NAVAL-SEM" \
    -srcfolder dist/NAVAL-SEM.app \
    -ov -format UDZO \
    dist/NAVAL-SEM.dmg
  echo " DMG  → dist/NAVAL-SEM.dmg (plain, install create-dmg for a nicer layout)"
fi

echo ""
echo " Done!"
echo "  dist/NAVAL-SEM.app"
echo "  dist/NAVAL-SEM.dmg"
echo ""
