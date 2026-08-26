@echo off
setlocal
REM Compile both Windows installers from the two .iss scripts:
REM   Output\MilanaSetup.exe              (models packed, optional task)
REM   Output\MilanaSetupNoOCRModels.exe   (data\models not captured)

set "SCRIPT_DIR=%~dp0"

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

echo === MilanaSetup.exe ===
"%ISCC%" "%SCRIPT_DIR%InnoSetupInstallerBuild.iss"
if errorlevel 1 exit /b 1

echo.
echo === MilanaSetupNoOCRModels.exe ===
"%ISCC%" "%SCRIPT_DIR%InnoSetupInstallerBuildNoOCRModels.iss"
if errorlevel 1 exit /b 1

echo.
echo Both installers written to:
echo   %SCRIPT_DIR%Output\MilanaSetup.exe
echo   %SCRIPT_DIR%Output\MilanaSetupNoOCRModels.exe
endlocal
