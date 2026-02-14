@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

REM === windows.bat ===
REM Creates ONLY launch_milana.cmd with venv activation + ui.py

echo ========================================
echo Milana Installer for Windows
echo ========================================
echo.

REM 1) Determine project root (absolute, without trailing backslash)
SET "SCRIPT_DIR=%~dp0"
SET "PROJECT_DIR=!SCRIPT_DIR!.."
:strip_trailing_slash
IF "!PROJECT_DIR:~-1!"=="\" SET "PROJECT_DIR=!PROJECT_DIR:~0,-1!" & goto strip_trailing_slash
echo Project directory: "!PROJECT_DIR!"
echo.

REM 2) Check Python
echo Checking Python installation...
where python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found in PATH!
    echo Please install Python 3.8+ and check "Add Python to PATH".
    pause
    exit /b 1
)
python --version
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to run Python
    pause
    exit /b 1
)
echo.

REM 3) Create / verify virtual environment
SET "VENV_DIR=!PROJECT_DIR!\mvenv"
echo Virtual environment: "!VENV_DIR!"
echo.

IF EXIST "!VENV_DIR!" (
    echo Virtual environment already exists.
    set /p CHOICE="Recreate? [y/N]: "
    echo.
    IF /I "!CHOICE!"=="y" (
        echo Removing old virtual environment...
        rmdir /s /q "!VENV_DIR!" 2>nul
        python -m venv "!VENV_DIR!"
        IF !ERRORLEVEL! NEQ 0 (
            echo ERROR: Failed to recreate virtual environment!
            pause
            exit /b 1
        )
        echo Virtual environment recreated.
    ) ELSE (
        echo Using existing virtual environment.
    )
) ELSE (
    echo Creating new virtual environment...
    python -m venv "!VENV_DIR!"
    IF !ERRORLEVEL! NEQ 0 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created.
)

IF NOT EXIST "!VENV_DIR!\Scripts\python.exe" (
    echo ERROR: python.exe not found in virtual environment!
    pause
    exit /b 1
)
echo Virtual environment verified.
echo.

REM 4) Install dependencies (direct venv python)
echo Upgrading pip...
"!VENV_DIR!\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1

IF NOT EXIST "!PROJECT_DIR!\install\requirements.txt" (
    echo ERROR: requirements.txt not found at "!PROJECT_DIR!\install\requirements.txt"!
    pause
    exit /b 1
)

echo Installing dependencies from requirements.txt...
echo This may take several minutes...
echo.
"!VENV_DIR!\Scripts\python.exe" -m pip install -r "!PROJECT_DIR!\install\requirements.txt"
IF !ERRORLEVEL! NEQ 0 (
    echo.
    echo WARNING: Some packages failed to install.
    set /p CHOICE="Continue? [Y/n]: "
    IF /I "!CHOICE!"=="n" pause & exit /b 1
)
echo.
echo Dependencies installed.
echo.

REM 5) Optional package checks
"!VENV_DIR!\Scripts\python.exe" -c "import torch; print('PyTorch:', torch.__version__)" 2>nul || echo PyTorch not found
"!VENV_DIR!\Scripts\python.exe" -c "import transformers; print('Transformers OK')" 2>nul || echo Transformers not found
echo.

REM ============= GENERATE LAUNCHER (line by line) =============
cd /d "!PROJECT_DIR!"

echo Creating launcher: launch_milana.cmd

REM Delete old launcher if exists
del launch_milana.cmd 2>nul

REM Write launcher line by line – 100% reliable
echo @echo off >> launch_milana.cmd
echo setlocal enabledelayedexpansion >> launch_milana.cmd
echo chcp 65001 ^>nul >> launch_milana.cmd
echo. >> launch_milana.cmd
echo echo ======================================== >> launch_milana.cmd
echo echo Starting Milana... >> launch_milana.cmd
echo echo ======================================== >> launch_milana.cmd
echo echo. >> launch_milana.cmd
echo. >> launch_milana.cmd
echo REM Get project root (where this .cmd resides) >> launch_milana.cmd
echo set "PROJECT_DIR=%%~dp0" >> launch_milana.cmd
echo if "!PROJECT_DIR:~-1!"=="\" set "PROJECT_DIR=!PROJECT_DIR:~0,-1!" >> launch_milana.cmd
echo. >> launch_milana.cmd
echo echo Project directory: "!PROJECT_DIR!" >> launch_milana.cmd
echo echo. >> launch_milana.cmd
echo. >> launch_milana.cmd
echo REM Activate virtual environment >> launch_milana.cmd
echo set "VENV_DIR=!PROJECT_DIR!\mvenv" >> launch_milana.cmd
echo if not exist "!VENV_DIR!\Scripts\activate.bat" ( >> launch_milana.cmd
echo     echo ERROR: Virtual environment not found at "!VENV_DIR!" >> launch_milana.cmd
echo     echo Please run windows.bat again to recreate it. >> launch_milana.cmd
echo     pause >> launch_milana.cmd
echo     exit /b 1 >> launch_milana.cmd
echo ) >> launch_milana.cmd
echo. >> launch_milana.cmd
echo echo Activating virtual environment... >> launch_milana.cmd
echo call "!VENV_DIR!\Scripts\activate.bat" >> launch_milana.cmd
echo if !ERRORLEVEL! NEQ 0 ( >> launch_milana.cmd
echo     echo ERROR: Failed to activate virtual environment! >> launch_milana.cmd
echo     pause >> launch_milana.cmd
echo     exit /b 1 >> launch_milana.cmd
echo ) >> launch_milana.cmd
echo. >> launch_milana.cmd
echo echo Running Milana UI... >> launch_milana.cmd
echo echo. >> launch_milana.cmd
echo python "!PROJECT_DIR!\ui.py" >> launch_milana.cmd
echo. >> launch_milana.cmd
echo if !ERRORLEVEL! NEQ 0 ( >> launch_milana.cmd
echo     echo. >> launch_milana.cmd
echo     echo Application closed with error code !ERRORLEVEL!. >> launch_milana.cmd
echo ) >> launch_milana.cmd
echo pause >> launch_milana.cmd

IF EXIST "launch_milana.cmd" (
    echo ✓ Launcher successfully created: "!PROJECT_DIR!\launch_milana.cmd"
) ELSE (
    echo ✗ ERROR: Failed to create launch_milana.cmd!
    echo   Check write permissions for "!PROJECT_DIR!".
    pause
    exit /b 1
)
echo.

REM 7) Final message
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.
echo To run Milana:
echo   Double-click "launch_milana.cmd" in the project folder.
echo.
echo Project folder: "!PROJECT_DIR!"
echo.
pause