@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

REM === windows.bat ===
REM Fixed version with better error handling and diagnostics

echo ========================================
echo Milana Installer for Windows
echo ========================================
echo.

REM 1) Определяем корневую директорию проекта
SET "SCRIPT_DIR=%~dp0"
echo Script directory: "!SCRIPT_DIR!"
SET "PROJECT_DIR=!SCRIPT_DIR!.."

REM Убираем возможный завершающий обратный слэш
IF "!PROJECT_DIR:~-1!"=="\" SET "PROJECT_DIR=!PROJECT_DIR:~0,-1!"
IF "!PROJECT_DIR:~-1!"=="\" SET "PROJECT_DIR=!PROJECT_DIR:~0,-1!"  ! Double-check
echo Project directory: "!PROJECT_DIR!"
echo.

REM 2) Проверяем Python с подробным выводом
echo Checking Python installation...
where python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found in PATH!
    echo.
    echo Please install Python 3.8+ from https://python.org
    echo And make sure to check "Add Python to PATH" during installation.
    echo.
    echo Current PATH: %PATH%
    pause
    exit /b 1
)

echo Python found in PATH.
python --version
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to run Python command
    pause
    exit /b 1
)
echo.

REM 3) Создаем виртуальное окружение
SET "VENV_DIR=!PROJECT_DIR!\mvenv"
echo Creating virtual environment at: "!VENV_DIR!"
echo.

IF EXIST "!VENV_DIR!" (
    echo Virtual environment already exists at "!VENV_DIR!"
    echo.
    set /p CHOICE="Recreate? [y/N]: "
    echo.
    IF /I "!CHOICE!"=="y" (
        echo Removing old virtual environment...
        rmdir /s /q "!VENV_DIR!" 2>nul
        python -m venv "!VENV_DIR!"
        IF !ERRORLEVEL! NEQ 0 (
            echo ERROR: Failed to recreate virtual environment!
            echo Please check if the folder is in use.
            pause
            exit /b 1
        )
        echo Virtual environment recreated successfully.
    ) ELSE (
        echo Using existing virtual environment.
    )
) ELSE (
    echo Creating new virtual environment...
    python -m venv "!VENV_DIR!"
    IF !ERRORLEVEL! NEQ 0 (
        echo ERROR: Failed to create virtual environment!
        echo.
        echo Possible causes:
        echo 1. Insufficient permissions
        echo 2. Antivirus blocking
        echo 3. Disk space issues
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)

REM Проверяем, что виртуальная среда создалась корректно
echo.
echo Verifying virtual environment...
IF NOT EXIST "!VENV_DIR!\Scripts\python.exe" (
    echo ERROR: python.exe not found in virtual environment!
    echo Virtual environment may be corrupted.
    dir "!VENV_DIR!\Scripts\" 2>nul || echo Cannot list Scripts directory
    pause
    exit /b 1
)

IF NOT EXIST "!VENV_DIR!\Scripts\pip.exe" (
    echo WARNING: pip.exe not found in virtual environment, but continuing...
)

echo Virtual environment verified: OK
echo.

REM 4) Устанавливаем зависимости БЕЗ активации (используем прямой путь к python из venv)
echo Installing dependencies using venv python...
echo.

REM Обновляем pip
echo Upgrading pip...
"!VENV_DIR!\Scripts\python.exe" -m pip install --upgrade pip
IF !ERRORLEVEL! NEQ 0 (
    echo WARNING: Failed to upgrade pip. Continuing...
)

REM Проверяем наличие requirements.txt
echo Checking for requirements.txt...
IF NOT EXIST "!PROJECT_DIR!\install\requirements.txt" (
    echo ERROR: requirements.txt not found at "!PROJECT_DIR!\install\requirements.txt"!
    echo Available files in install directory:
    dir "!PROJECT_DIR!\install\" 2>nul || echo Install directory not found
    pause
    exit /b 1
)

echo requirements.txt found. Installing dependencies...
echo This may take several minutes...
echo.

"!VENV_DIR!\Scripts\python.exe" -m pip install -r "!PROJECT_DIR!\install\requirements.txt"
IF !ERRORLEVEL! NEQ 0 (
    echo ERROR: Failed to install some dependencies!
    echo.
    echo Possible solutions:
    echo 1. Check your internet connection
    echo 2. Try running: "!VENV_DIR!\Scripts\python.exe" -m pip install -r "!PROJECT_DIR!\install\requirements.txt" manually
    echo 3. Some packages might require additional system dependencies
    echo.
    echo Continue anyway? Some features might not work.
    echo.
    set /p CHOICE="Continue? [Y/n]: "
    IF /I NOT "!CHOICE!"=="n" (
        echo Continuing despite errors...
    ) ELSE (
        pause
        exit /b 1
    )
)

echo.
echo Dependencies installed successfully!
echo.

REM 5) Проверяем установку основных пакетов
echo Verifying key packages...
"!VENV_DIR!\Scripts\python.exe" -c "import sys; print('Python version:', sys.version)"
"!VENV_DIR!\Scripts\python.exe" -c "import torch; print('PyTorch version:', torch.__version__)" 2>nul && echo "PyTorch: OK" || echo "PyTorch: Not installed"
"!VENV_DIR!\Scripts\python.exe" -c "import transformers; print('Transformers: OK')" 2>nul || echo "Transformers: Not installed"
echo.

REM 6) Создаем скрипт запуска (БЕЗ активации venv)
echo Creating launcher script...
(
echo @echo off
echo setlocal enabledelayedexpansion
echo chcp 65001 ^>nul
echo.
echo echo ========================================
echo echo Starting Milana...
echo echo ========================================
echo echo.
echo 
echo set "PROJECT_DIR=%%~dp0"
echo if "!PROJECT_DIR:~-1!"=="\" set "PROJECT_DIR=!PROJECT_DIR:~0,-1!"
echo.
echo echo Project directory: "!PROJECT_DIR!"
echo echo.
echo 
echo REM Используем прямой путь к Python из виртуальной среды
echo set "PYTHON_EXE=!PROJECT_DIR!\mvenv\Scripts\python.exe"
echo.
echo if not exist "!PYTHON_EXE!" (
echo     echo ERROR: Python not found at: "!PYTHON_EXE!"
echo     echo.
echo     echo Please run windows.bat again to recreate virtual environment.
echo     pause
echo     exit /b 1
echo )
echo.
echo echo Running Milana UI...
echo echo.
echo "!PYTHON_EXE!" "!PROJECT_DIR!\ui.py"
echo.
echo if !ERRORLEVEL! NEQ 0 (
echo     echo.
echo     echo Application closed with error.
echo )
echo pause
) > "!PROJECT_DIR!\run_ui.cmd"

echo Launcher created: "!PROJECT_DIR!\run_ui.cmd"
echo.

REM 7) Создаем простой ярлык (без PowerShell)
echo Creating desktop shortcut...
(
echo @echo off
echo start "" "%%~dp0run_ui.cmd"
) > "!PROJECT_DIR!\start_milana.cmd"

echo You can create a shortcut manually:
echo 1. Right-click on "run_ui.cmd"
echo 2. Select "Create shortcut"
echo 3. Move shortcut to Desktop
echo.

REM 8) Финальные инструкции
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.
echo To run Milana:
echo.
echo Option 1: Double-click "run_ui.cmd" in project folder
echo.
echo Option 2: Create shortcut manually (see above)
echo.
echo Project location: "!PROJECT_DIR!"
echo Virtual environment: "!VENV_DIR!"
echo.
echo Troubleshooting:
echo 1. If Milana doesn't start, check if Python 3.8+ is installed
echo 2. Run windows.bat again if virtual environment is corrupted
echo 3. Check install/requirements.txt for package versions
echo.
pause