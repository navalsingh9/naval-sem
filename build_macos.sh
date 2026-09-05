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
  uv sync --locked --group build
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

# naval_sem.spec is shared across all 3 platforms and now includes a
# platform-guarded BUNDLE() step (see naval_sem.spec) so that on macOS it
# produces a real dist/NAVAL-SEM.app, not just the plain dist/NAVAL-SEM
# binary EXE() alone would give you. The fallback below only exists as a
# safety net for cases where that still fails for some other reason.
if [ ! -d "dist/NAVAL-SEM.app" ]; then
  echo " No .app found from the spec — rebuilding with --windowed..."
  echo " WARNING: this fallback path is not kept in sync with naval_sem.spec's"
  echo " hiddenimports/datas and has previously shipped broken builds. Prefer"
  echo " fixing the spec over relying on this."
  if command -v uv &>/dev/null; then
    uv run pyinstaller launcher.py \
      --name "NAVAL-SEM" \
      --windowed \
      --clean --noconfirm \
      --osx-bundle-identifier "io.naval-sem.app" \
      --add-data "static:static" \
      --add-data "app:app" \
      --add-data "fonts:fonts" \
      --hidden-import "fastapi" \
      --hidden-import "starlette" \
      --hidden-import "uvicorn" \
      --hidden-import "uvicorn.logging" \
      --hidden-import "uvicorn.loops.auto" \
      --hidden-import "uvicorn.protocols.http.auto" \
      --hidden-import "uvicorn.protocols.websockets.auto" \
      --hidden-import "uvicorn.lifespan.on" \
      --hidden-import "pydantic" \
      --hidden-import "pydantic_core" \
      --hidden-import "multipart" \
      --hidden-import "semopy" \
      --hidden-import "scipy" \
      --hidden-import "numpy" \
      --hidden-import "pandas" \
      --hidden-import "pyreadstat" \
      --hidden-import "webview" \
      --hidden-import "h11" \
      --hidden-import "anyio._backends._asyncio" \
      --hidden-import "email_validator" \
      --hidden-import "reportlab" \
      --hidden-import "app.main"
  else
    pyinstaller launcher.py \
      --name "NAVAL-SEM" \
      --windowed \
      --clean --noconfirm \
      --osx-bundle-identifier "io.naval-sem.app" \
      --add-data "static:static" \
      --add-data "app:app" \
      --add-data "fonts:fonts" \
      --hidden-import "app.main"
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
