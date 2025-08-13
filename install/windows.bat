@echo off
REM === install\windows.bat ===

REM 1) Determine project root and locate Python
SETLOCAL ENABLEDELAYEDEXPANSION
REM %~dp0\.. points to the parent folder (project root) where data/ and ui.py reside
SET "PROJECT_DIR=%~dp0..\"
REM Remove trailing backslash
SET "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

REM 2) Create a virtual environment 'mvenv' in project root
"%~dp0..\python" -m venv "%PROJECT_DIR%\mvenv" 2>nul || (
  echo Warning: python.exe not found next to this script; falling back to PATH
  python -m venv "%PROJECT_DIR%\mvenv"
)

REM 3) Activate venv and install dependencies
call "%PROJECT_DIR%\mvenv\Scripts\activate.bat"
pip install --upgrade pip
pip install -r "%PROJECT_DIR%\install\requirements.txt"

REM 4) Generate a launcher script run_ui.cmd in project root
(
  echo @echo off
  echo call "%%~dp0\mvenv\Scripts\activate.bat"
  echo python "%%~dp0\ui.py"
) > "%PROJECT_DIR%\run_ui.cmd"
echo Created launcher: %PROJECT_DIR%\run_ui.cmd

REM 5) Create shortcuts (Milana.lnk) on Desktop and in project root
for %%L in (
  "%USERPROFILE%\Desktop\Milana.lnk"
  "%PROJECT_DIR%\Milana.lnk"
) do (
  powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell; ^
     $sc = $ws.CreateShortcut('%%~L'); ^
     $sc.TargetPath     = '%PROJECT_DIR%\run_ui.cmd'; ^
     $sc.WorkingDirectory = '%PROJECT_DIR%'; ^
     $sc.IconLocation   = '%PROJECT_DIR%\data\icons\icon.ico'; ^
     $sc.Save()"
  echo Shortcut created: %%L
)

echo.
echo Installation complete. You can launch Milana from your Desktop or the project folder.
pause
ENDLOCAL
