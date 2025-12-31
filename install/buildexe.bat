@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo Milana Build Script
echo ========================================

REM 1) Define paths (safe with spaces)
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
IF "!PROJECT_ROOT:~-1!"=="\" SET "PROJECT_ROOT=!PROJECT_ROOT:~0,-1!"
set "VENV_DIR=%PROJECT_ROOT%\mvenv"
set "REQUIREMENTS=%SCRIPT_DIR%requirements.txt"

REM 2) Clean previous builds
echo Cleaning previous builds...
if exist "%PROJECT_ROOT%\build" rmdir /s /q "%PROJECT_ROOT%\build"
if exist "%PROJECT_ROOT%\dist" rmdir /s /q "%PROJECT_ROOT%\dist"
if exist "%PROJECT_ROOT%\Milana.exe" del "%PROJECT_ROOT%\Milana.exe"
if exist "%PROJECT_ROOT%\_internal" rmdir /s /q "%PROJECT_ROOT%\_internal"

REM 3) Check virtual environment - НЕ ПЕРЕСОЗДАЕМ если существует
echo Checking virtual environment in "%VENV_DIR%"...
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    
    REM Try python3 first, then python
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python not found! Install Python 3.8+ first.
        pause
        exit /b 1
    )
    
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        echo Please check Python installation.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists. Using existing one.
)

REM 4) Activate virtual environment using DIRECT python path
echo Activating virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ERROR: activate.bat not found! Virtual environment may be corrupted.
    echo Please delete "%VENV_DIR%" folder and run this script again.
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    echo Trying alternative activation method...
    
    REM Use direct python from venv
    if exist "%VENV_DIR%\Scripts\python.exe" (
        echo Using venv python directly...
        set "PATH=%VENV_DIR%\Scripts;%PATH%"
    ) else (
        echo FATAL: No python.exe in virtual environment!
        echo Please delete "%VENV_DIR%" folder and run this script again.
        pause
        exit /b 1
    )
)

REM 5) Update pip and install dependencies
echo Updating pip...
python -m pip install --upgrade pip

if exist "%REQUIREMENTS%" (
    echo Installing dependencies from requirements.txt...
    pip install -r "%REQUIREMENTS%"
    
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies!
        echo Please check your internet connection and requirements.txt
        pause
        exit /b 1
    )
) else (
    echo ERROR: requirements.txt file not found at "%REQUIREMENTS%"!
    pause
    exit /b 1
)

REM 6) Check for ui.py
if not exist "%PROJECT_ROOT%\ui.py" (
    echo ERROR: ui.py file not found in project root!
    pause
    exit /b 1
)

REM 7) Build with PyInstaller
echo ========================================
echo Building executable...
echo ========================================

REM Change to project root directory
cd /d "%PROJECT_ROOT%"

REM Build command (paths in quotes for spaces)
pyinstaller --onedir --icon="data\icons\icon.ico" --name "Milana" ^
--clean ^
--distpath "%PROJECT_ROOT%\dist" ^
--workpath "%PROJECT_ROOT%\build" ^
--collect-all easyocr ^
--hidden-import chromadb.db.duckdb ^
--hidden-import chromadb.telemetry.product.posthog ^
--hidden-import chromadb.telemetry.opentelemetry ^
--hidden-import chromadb.api.rust ^
--hidden-import transformers ^
--hidden-import torch ^
--hidden-import sentencepiece ^
--hidden-import tokenizers ^
--hidden-import accelerate ^
--hidden-import huggingface_hub ^
--hidden-import paddle ^
--hidden-import cv2 ^
--hidden-import sklearn.utils._weight_vector ^
--hidden-import sklearn.neighbors._typedefs ^
--hidden-import sklearn.neighbors._quad_tree ^
--hidden-import scipy._lib.messagestream ^
--clean ^
--noconfirm ^
--noconsole ^
"ui.py"

if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

REM 8) Move built files to project root
echo Moving built files to project root...
if exist "%PROJECT_ROOT%\dist\Milana\*" (
    echo Copying files from "%PROJECT_ROOT%\dist\Milana\" to "%PROJECT_ROOT%\"
    xcopy /E /I /Y "%PROJECT_ROOT%\dist\Milana\*" "%PROJECT_ROOT%\"
    
    REM Rename the main folder from Milana to _internal
    if exist "%PROJECT_ROOT%\Milana" (
        rename "%PROJECT_ROOT%\Milana" "_internal"
        echo ✓ Renamed Milana folder to _internal
    )
) else (
    echo ERROR: No files found in "%PROJECT_ROOT%\dist\Milana\"
    echo Checking dist folder contents:
    dir "%PROJECT_ROOT%\dist"
    pause
    exit /b 1
)

REM 9) Clean temporary files
echo Cleaning temporary files...
if exist "%PROJECT_ROOT%\build" rmdir /s /q "%PROJECT_ROOT%\build"
if exist "%PROJECT_ROOT%\dist" rmdir /s /q "%PROJECT_ROOT%\dist"
if exist "%PROJECT_ROOT%\Milana.spec" del "%PROJECT_ROOT%\Milana.spec"

REM 10) Verify build contents in project root
echo Verifying build in project root...
if exist "%PROJECT_ROOT%\Milana.exe" (
    echo ✓ Executable file: "%PROJECT_ROOT%\Milana.exe"
    for %%F in ("%PROJECT_ROOT%\Milana.exe") do echo ✓ File size: %%~zF bytes
) else (
    echo ✗ ERROR: Milana.exe not found in project root!
    echo Current files in root:
    dir "%PROJECT_ROOT%\"
)

if exist "%PROJECT_ROOT%\_internal" (
    echo ✓ _internal folder: "%PROJECT_ROOT%\_internal"
    dir "%PROJECT_ROOT%\_internal" | find "File(s)" >nul && (
        echo ✓ _internal folder contains files
        echo ✓ First few files:
        dir "%PROJECT_ROOT%\_internal" /b 2>nul | findstr /v "Directory" | head -5
    )
) else (
    echo ✗ ERROR: _internal folder not found in project root!
)

REM 11) Create or check launcher.py
echo Checking launcher.py...
if not exist "%PROJECT_ROOT%\launcher.py" (
    echo Creating minimal launcher.py...
    (
        echo import os
        echo import sys
        echo import subprocess
        echo.
        echo # Get current directory
        echo current_dir = os.path.dirname(os.path.abspath(__file__))
        echo.
        echo # Run the main executable
        echo exe_path = os.path.join(current_dir, "Milana.exe")
        echo.
        echo if os.path.exists(exe_path):
        echo     subprocess.run([exe_path] + sys.argv[1:])
        echo else:
        echo     print("Error: Milana.exe not found!")
        echo     input("Press Enter to exit...")
    ) > "%PROJECT_ROOT%\launcher.py"
    echo ✓ Created launcher.py
) else (
    echo ✓ launcher.py already exists
)

REM 12) Final message
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo To run Milana:
echo 1. Double-click "Milana.exe" in project folder
echo 2. Or run "launcher.py"
echo.
echo Project location: "%PROJECT_ROOT%"
echo.
pause