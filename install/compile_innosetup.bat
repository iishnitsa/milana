@echo off
setlocal
REM Compile both Windows installers:
REM   Output\MilanaSetup.exe           (models packed, optional task)
REM   Output\MilanaSetup-nomodels.exe  (no weights, no prompt)

set "SCRIPT_DIR=%~dp0"
set "ISS=%SCRIPT_DIR%InnoSetupInstallerBuild.iss"

set "ISCC="
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles(x86)%\Inno Setup 5\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 5\ISCC.exe"
where iscc >nul 2>&1 && for /f "delims=" %%I in ('where iscc') do set "ISCC=%%I"

if not defined ISCC (
    echo ERROR: Inno Setup Compiler ^(ISCC.exe^) not found.
    echo Install Inno Setup 6 or add ISCC.exe to PATH.
    exit /b 1
)

echo Using: %ISCC%
echo.

echo === MilanaSetup.exe (with optional image models) ===
"%ISCC%" "%ISS%"
if errorlevel 1 exit /b 1

echo.
echo === MilanaSetup-nomodels.exe (no models in the installer) ===
"%ISCC%" /DNoModels "%ISS%"
if errorlevel 1 exit /b 1

echo.
echo Both installers written to:
echo   %SCRIPT_DIR%Output\MilanaSetup.exe
echo   %SCRIPT_DIR%Output\MilanaSetup-nomodels.exe
endlocal
