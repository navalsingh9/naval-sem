#!/usr/bin/env bash
# ============================================================
#  NAVAL-SEM — macOS Build Script
#  Produces:
#    dist/NAVAL-SEM.app   (double-clickable app bundle)
#    dist/NAVAL-SEM.dmg   (drag-to-Applications installer)
#
#  Prerequisites:
#    uv sync
#    brew install create-dmg
# ============================================================

set -e
cd "$(dirname "$0")"

if [ "$(uname)" != "Darwin" ]; then
  echo " This script builds a macOS .app/.dmg and must be run on macOS."
  exit 1
fi

echo ""
echo " ========================================="
echo "  NAVAL-SEM macOS Build"
echo " ========================================="
echo ""

# ── 1. Dependencies ────────────────────────────────────────────────────────
echo " [1/3] Verifying build environment..."
if command -v uv &>/dev/null; then
  echo " Found 'uv'. Synchronizing dependencies from secure lockfile..."
  uv sync --locked
else
  echo " 'uv' not found. Proceeding with active environment packages..."
fi

if ! command -v create-dmg &>/dev/null; then
  echo ""
  echo " create-dmg not found. Install it first:"
  echo "   brew install create-dmg"
  exit 1
fi

# ── 2. .app bundle ───────────────────────────────────────────────────────────
echo " [2/3] Building .app with PyInstaller..."
rm -rf dist/NAVAL-SEM.app dist/NAVAL-SEM   # clear stale output so the check below can't be fooled

if command -v uv &>/dev/null; then
  uv run pyinstaller naval_sem.spec --clean --noconfirm
else
  pyinstaller naval_sem.spec --clean --noconfirm
fi

# naval_sem.spec is shared across all 3 platforms and only defines an EXE()
# (single-file) target, not a BUNDLE() — so on macOS it produces a plain
# dist/NAVAL-SEM binary, never dist/NAVAL-SEM.app. This mirrors the same
# fallback the CI job already relies on (.github/workflows/release.yml,
# build-macos job): rebuild straight from launcher.py with --windowed,
# which does produce a real .app bundle.
if [ ! -d "dist/NAVAL-SEM.app" ]; then
  echo " No .app found from the spec — rebuilding with --windowed..."
  if command -v uv &>/dev/null; then
    uv run pyinstaller launcher.py \
      --name "NAVAL-SEM" \
      --windowed \
      --clean --noconfirm \
      --osx-bundle-identifier "io.naval-sem.app"
  else
    pyinstaller launcher.py \
      --name "NAVAL-SEM" \
      --windowed \
      --clean --noconfirm \
      --osx-bundle-identifier "io.naval-sem.app"
  fi
fi

if [ ! -d "dist/NAVAL-SEM.app" ]; then
  echo " dist/NAVAL-SEM.app still missing after fallback build — aborting."
  exit 1
fi
echo " .app  → dist/NAVAL-SEM.app"

# ── 3. DMG ────────────────────────────────────────────────────────────────────
echo " [3/3] Creating DMG..."
rm -f dist/NAVAL-SEM.dmg   # create-dmg refuses to overwrite an existing file

create-dmg \
  --volname "NAVAL-SEM" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --app-drop-link 425 190 \
  dist/NAVAL-SEM.dmg \
  "dist/NAVAL-SEM.app"

echo ""
echo " Done!"
echo "  dist/NAVAL-SEM.app"
echo "  dist/NAVAL-SEM.dmg"
echo ""
echo " No Apple Developer ID is used, so Gatekeeper will block the first"
echo " launch — right-click the app → Open to bypass it."
echo ""
