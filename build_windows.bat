@echo off
REM ============================================================
REM  NAVAL-SEM  -  Windows Build Script
REM  Produces:
REM    dist\NAVAL-SEM.exe          (portable single-file EXE)
REM    dist\NAVAL-SEM-Setup.msi   (installer, requires WiX 3)
REM
REM  Prerequisites (install once):
REM    pip install pyinstaller pywebview
REM    WiX Toolset 3.x  https://github.com/wixtoolset/wix3/releases
REM ============================================================

setlocal EnableDelayedExpansion

echo.
echo  =========================================
echo   NAVAL-SEM Build Script
echo  =========================================
echo.

REM -- Add WiX 3 to PATH automatically -----------------------------------
set "WIX3_BIN=C:\Program Files (x86)\WiX Toolset v3.14\bin"
if exist "%WIX3_BIN%\candle.exe" (
    set "PATH=%PATH%;%WIX3_BIN%"
)

REM -- 1. Check Python ---------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install Python 3.10+ and add to PATH.
    exit /b 1
)

REM -- 2. Install / upgrade dependencies ---------------------------------
echo  [1/4] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ERROR] pip install failed.
    exit /b 1
)

REM -- 3. Build EXE with PyInstaller -------------------------------------
echo  [2/4] Building EXE with PyInstaller...
pyinstaller naval_sem.spec --clean --noconfirm
if errorlevel 1 (
    echo  [ERROR] PyInstaller build failed.
    exit /b 1
)
echo  EXE -^> dist\NAVAL-SEM.exe

REM -- 4. Build MSI with WiX 3 ------------------------------------------
echo  [3/4] Checking for WiX toolset...

candle.exe --version >nul 2>&1
if not errorlevel 1 (
    echo  Using WiX 3...
    call :build_msi_wix3
    goto :done
)

echo  [WARN] WiX 3 toolset not found. Skipping MSI build.
echo         Install WiX 3 from https://github.com/wixtoolset/wix3/releases
echo         Then re-run this script to also produce an MSI.
goto :done

:build_msi_wix3
candle.exe installer\naval_sem.wxs -o installer\naval_sem.wixobj
if errorlevel 1 goto :wix_error
light.exe installer\naval_sem.wixobj -o dist\NAVAL-SEM-Setup.msi -ext WixUIExtension
if errorlevel 1 goto :wix_error
echo  MSI  -^> dist\NAVAL-SEM-Setup.msi
exit /b 0

:wix_error
echo  [ERROR] WiX build failed. Check installer\naval_sem.wxs
exit /b 1

:done
echo.
echo  [4/4] Done!
echo.
echo  Outputs:
if exist "dist\NAVAL-SEM.exe"       echo    dist\NAVAL-SEM.exe
if exist "dist\NAVAL-SEM-Setup.msi" echo    dist\NAVAL-SEM-Setup.msi
echo.
pause
