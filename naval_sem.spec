# naval_sem.spec
# PyInstaller spec for NAVAL-SEM
# Usage: pyinstaller naval_sem.spec

import sys
from pathlib import Path

block_cipher = None

# Collect all hidden imports that PyInstaller misses for these packages
hiddenimports = [
    # FastAPI / Starlette
    "fastapi", "fastapi.middleware", "fastapi.middleware.cors",
    "fastapi.responses", "fastapi.staticfiles",
    "starlette", "starlette.middleware", "starlette.middleware.cors",
    "starlette.middleware.base", "starlette.staticfiles", "starlette.responses",
    # Uvicorn
    "uvicorn", "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.loops.asyncio", "uvicorn.protocols", "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on",
    # Pydantic
    "pydantic", "pydantic.v1", "pydantic_core",
    # Python multipart
    "multipart",
    # semopy and its deps
    "semopy", "semopy.model", "semopy.stats",
    # scipy, numpy, pandas
    "scipy", "scipy.optimize", "scipy.linalg", "scipy.stats",
    "numpy", "pandas",
    "openpyxl", "xlrd",
    # pyreadstat (SPSS)
    "pyreadstat",
    # pywebview
    "webview",
    # h11 (uvicorn http)
    "h11",
    # anyio
    "anyio", "anyio._backends._asyncio",
    # email validator used by pydantic
    "email_validator",
    # ReportLab
    "reportlab",
    "reportlab.platypus",
    "reportlab.platypus.flowables",
    "reportlab.graphics.shapes",
    "reportlab.graphics.charts.barcharts",
    "reportlab.pdfbase.ttfonts",
    "reportlab.lib.styles",
    "PIL", "PIL.Image",   # Pillow — reportlab dependency
]

a = Analysis(
    ["launcher.py"],
    pathex=[str(Path(".").resolve())],
    binaries=[],
    datas=[
        # Bundle the static HTML frontend
        ("static", "static"),
        # Bundle the app package
        ("app", "app"),
        # DejaVu fonts for Unicode PDF output (Greek, arrows, checkmarks)
        # Bundled in repo under fonts/ — works on Windows, macOS and Linux CI
        *[
            (str(p), "fonts")
            for p in Path("fonts").glob("*.ttf")
        ],
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy stuff we don't need
        "tkinter", "matplotlib", "IPython",
        "jupyter", "notebook", "test", "tests",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── Single-file EXE (--onefile) ───────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="NAVAL-SEM",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,           # compress with UPX if installed (reduces size ~30%)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,      # no console window — set True while debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Windows icon (place a naval_sem.ico in the project root)
    icon="naval_sem.ico" if Path("naval_sem.ico").exists() else None,
)
