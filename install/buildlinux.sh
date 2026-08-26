#!/usr/bin/env bash
# === buildlinux.sh ===
# Milana Build Script for Linux (pyenv version)
# Ported from buildexe.bat - with CPU version of PyTorch to match Windows size

set -euo pipefail

echo "========================================"
echo "Milana Build Script"
echo "========================================"

# Determine paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Define paths
VENV_DIR="$PROJECT_ROOT/mvenv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
PYTHON_VERSION="3.13.7"

# Check pyenv
echo "Checking for pyenv..."
if ! command -v pyenv &> /dev/null; then
    echo "ERROR: pyenv is not installed."
    echo "Please install pyenv first: https://github.com/pyenv/pyenv-installer"
    exit 1
fi
echo "✓ pyenv found: $(which pyenv)"

# Install Python via pyenv if needed
echo "Checking for Python $PYTHON_VERSION via pyenv..."
if pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
    echo "Python $PYTHON_VERSION already installed."
else
    echo "Python $PYTHON_VERSION not found. Installing (this may take a while)..."
    pyenv install "$PYTHON_VERSION"
fi

PYENV_ROOT="$(pyenv root)"
PYTHON_CMD="$PYENV_ROOT/versions/$PYTHON_VERSION/bin/python3.13"
if [ ! -f "$PYTHON_CMD" ]; then
    echo "ERROR: Cannot find Python $PYTHON_VERSION at $PYTHON_CMD"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION found: $PYTHON_CMD"

# Create/verify virtual environment (do NOT recreate if exists)
echo "Checking virtual environment in \"$VENV_DIR\"..."
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Using existing one."
else
    echo "Creating virtual environment with Python $PYTHON_VERSION..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment!"
        exit 1
    fi
    echo "Virtual environment created successfully."
fi

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "ERROR: Virtual environment creation failed!"
    exit 1
fi
echo "✓ Virtual environment verified."

# Activate
source "$VENV_DIR/bin/activate"

# Update pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install all dependencies from requirements.txt
if [ ! -f "$REQUIREMENTS" ]; then
    echo "ERROR: requirements.txt not found at \"$REQUIREMENTS\"!"
    exit 1
fi
echo "Installing remaining dependencies from requirements.txt..."
pip install -r "$REQUIREMENTS"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies!"
    echo "Please check your internet connection and requirements.txt"
    exit 1
fi

# Check launcher.py
if [ ! -f "$PROJECT_ROOT/launcher.py" ]; then
    echo "ERROR: launcher.py not found in project root!"
    exit 1
fi

# 11) Ensure PyInstaller
pip install pyinstaller --quiet

# Clean previous builds (safe, after environment is ready)
echo "Cleaning previous builds..."
rm -rf "$PROJECT_ROOT/build" "$PROJECT_ROOT/dist" \
       "$PROJECT_ROOT/Milana" "$PROJECT_ROOT/_internal" \
       "$PROJECT_ROOT/Milana.spec"

# Build with PyInstaller
echo "========================================"
echo "Building executable..."
echo "========================================"

cd "$PROJECT_ROOT"

# Tcl/Tk data dirs (PyInstaller hook sometimes ships only _tcl_data → runtime crash on _tk_data)
TCL_DIR=""
TK_DIR=""
for d in /usr/lib/tcl8.6 /usr/share/tcltk/tcl8.6; do
    if [ -f "$d/init.tcl" ]; then TCL_DIR="$d"; break; fi
done
for d in /usr/lib/tk8.6 /usr/share/tcltk/tk8.6; do
    if [ -f "$d/tk.tcl" ]; then TK_DIR="$d"; break; fi
done
ADD_DATA_ARGS=()
if [ -n "$TCL_DIR" ]; then
    echo "✓ Tcl data: $TCL_DIR → _tcl_data"
    ADD_DATA_ARGS+=(--add-data "$TCL_DIR:_tcl_data")
else
    echo "WARNING: Tcl data dir not found (install tk/tcl packages)"
fi
if [ -n "$TK_DIR" ]; then
    echo "✓ Tk data: $TK_DIR → _tk_data"
    ADD_DATA_ARGS+=(--add-data "$TK_DIR:_tk_data")
else
    echo "WARNING: Tk data dir not found (install tk/tcl packages)"
fi

pyinstaller --onedir --icon="data/icons/icon.ico" --name "Milana" \
--distpath "$PROJECT_ROOT/dist" \
--workpath "$PROJECT_ROOT/build" \
--strip \
--python-option=OO \
--collect-data ddgs \
--collect-data fake_useragent \
--collect-all fake_useragent \
--collect-all easyocr \
--collect-all ddgs \
--collect-all cloudscraper \
--hidden-import cloudscraper \
--hidden-import chromadb.db.duckdb \
--hidden-import chromadb.telemetry.product.posthog \
--hidden-import chromadb.telemetry.opentelemetry \
--hidden-import chromadb.api.rust \
--hidden-import transformers \
--hidden-import torch \
--hidden-import sentencepiece \
--hidden-import tokenizers \
--hidden-import accelerate \
--hidden-import huggingface_hub \
--hidden-import paddle \
--hidden-import cv2 \
--hidden-import sklearn.utils._weight_vector \
--hidden-import sklearn.neighbors._typedefs \
--hidden-import sklearn.neighbors._quad_tree \
--hidden-import scipy._lib.messagestream \
--hidden-import cryptography \
--hidden-import cryptography.fernet \
--hidden-import cryptography.hazmat.primitives \
--hidden-import cryptography.hazmat.primitives.kdf.pbkdf2 \
--hidden-import cryptography.hazmat.backends \
--collect-all dulwich \
--hidden-import dulwich.porcelain \
--hidden-import dulwich.objects \
--hidden-import dulwich.repo \
"${ADD_DATA_ARGS[@]}" \
--noconsole \
--clean \
--noconfirm \
launcher.py

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi

# Move built files
echo "Moving built files to project root..."
if [ -d "$PROJECT_ROOT/dist/Milana" ]; then
    if [ -f "$PROJECT_ROOT/dist/Milana/Milana" ]; then
        mv "$PROJECT_ROOT/dist/Milana/Milana" "$PROJECT_ROOT/"
        echo "✓ Moved executable"
    fi
    if [ -d "$PROJECT_ROOT/dist/Milana/_internal" ]; then
        rm -rf "$PROJECT_ROOT/_internal"
        mv "$PROJECT_ROOT/dist/Milana/_internal" "$PROJECT_ROOT/"
        echo "✓ Moved _internal folder"
    fi
else
    echo "ERROR: Build output not found!"
    exit 1
fi

# Safety net: ensure _tk_data / _tcl_data exist (hook or --add-data may have missed)
ensure_tk_data() {
    local dest="$PROJECT_ROOT/_internal"
    if [ ! -d "$dest/_tcl_data" ] && [ -n "$TCL_DIR" ]; then
        echo "Copying Tcl data → _internal/_tcl_data"
        mkdir -p "$dest/_tcl_data"
        cp -a "$TCL_DIR"/. "$dest/_tcl_data/"
    fi
    if [ ! -d "$dest/_tk_data" ] && [ -n "$TK_DIR" ]; then
        echo "Copying Tk data → _internal/_tk_data"
        mkdir -p "$dest/_tk_data"
        cp -a "$TK_DIR"/. "$dest/_tk_data/"
    fi
    if [ -d "$dest/_tcl_data" ] && [ -d "$dest/_tk_data" ]; then
        echo "✓ Tcl/Tk data present under _internal"
    else
        echo "WARNING: _tcl_data / _tk_data still missing — ./Milana will fail at pyi_rth__tkinter"
    fi
}
ensure_tk_data

# Clean temporary build folders
rm -rf "$PROJECT_ROOT/build" "$PROJECT_ROOT/dist" "$PROJECT_ROOT/Milana.spec"

# Verify
echo "========================================"
echo "Build completed!"
echo "========================================"
if [ -f "$PROJECT_ROOT/Milana" ]; then
    echo "✓ Executable: $PROJECT_ROOT/Milana"
    echo "✓ _internal size: $(du -sh "$PROJECT_ROOT/_internal" 2>/dev/null | cut -f1)"
else
    echo "✗ Executable not found!"
fi
echo ""
echo "Run with: ./Milana"