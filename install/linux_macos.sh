#!/usr/bin/env bash
# === linux_macos.sh ===
# Clean version with proper read -r and explicit python3.8+ (<=3.13)

set -e

# 1) Determine project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# 2) Explicitly require Python 3.8 - 3.13
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found! Install Python 3.8-3.13 first."
    echo ""
    echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  Fedora: sudo dnf install python3 python3-virtualenv"
    echo "  macOS: brew install python@3.11"
    echo "  Or download from https://python.org"
    exit 1
fi

# Check Python version (must be 3.8 - 3.13)
PYTHON_CMD="python3"
PY_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ "$PY_MAJOR" -ne 3 ] || [ "$PY_MINOR" -lt 8 ] || [ "$PY_MINOR" -gt 13 ]; then
    echo "ERROR: Python 3.8 - 3.13 required, but found $PY_VERSION"
    echo ""
    echo "Current Python: $(which $PYTHON_CMD)"
    echo ""
    echo "Please install Python 3.8 - 3.13 and ensure it's the default python3."
    exit 1
fi

echo "✓ Python $PY_VERSION found: $(which $PYTHON_CMD)"
echo ""

# 3) Create/verify virtual environment
VENV_PATH="$PROJECT_ROOT/mvenv"

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists at $VENV_PATH"
    printf "Recreate? (y/N): "
    read -r yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        echo "Removing old virtual environment..."
        rm -rf "$VENV_PATH"
        echo "Creating new virtual environment..."
        $PYTHON_CMD -m venv "$VENV_PATH"
    else
        echo "Using existing virtual environment."
    fi
else
    echo "Creating virtual environment at $VENV_PATH..."
    $PYTHON_CMD -m venv "$VENV_PATH"
fi

# Verify virtual environment
if [ ! -f "$VENV_PATH/bin/python" ]; then
    echo "ERROR: Virtual environment creation failed!"
    exit 1
fi
echo "✓ Virtual environment verified."
echo ""

# 4) Activate venv and install dependencies
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "✓ Python in venv: $(which python)"
echo ""

echo "Upgrading pip..."
pip install --upgrade pip

# Check requirements file
REQUIREMENTS_FILE="$PROJECT_ROOT/install/requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "ERROR: requirements.txt not found at $REQUIREMENTS_FILE"
    exit 1
fi

echo "Installing dependencies from requirements.txt..."
echo "This may take several minutes..."
echo ""
pip install -r "$REQUIREMENTS_FILE"

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Some packages failed to install."
    printf "Continue anyway? [Y/n]: "
    read -r cont
    if [[ "$cont" =~ ^[Nn]$ ]]; then
        exit 1
    fi
fi
echo "✓ Dependencies installed."
echo ""

# 5) Verify key packages
echo "Verifying key packages..."
python -c "import torch; print('  PyTorch:', torch.__version__)" 2>/dev/null || echo "  PyTorch: Not installed"
python -c "import transformers; print('  Transformers: OK')" 2>/dev/null || echo "  Transformers: Not installed"
python -c "import easyocr; print('  EasyOCR: OK')" 2>/dev/null || echo "  EasyOCR: Not installed"
python -c "import customtkinter; print('  CustomTkinter: OK')" 2>/dev/null || echo "  CustomTkinter: Not installed"
echo ""

# 6) Create launcher script
echo "Creating launcher: run_milana.sh"
LAUNCHER="$PROJECT_ROOT/run_milana.sh"

cat > "$LAUNCHER" << 'EOF'
#!/usr/bin/env bash
# === run_milana.sh ===
# Launcher with venv activation

set -e

# Get project root (where this script resides)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "Starting Milana..."
echo "========================================"
echo ""
echo "Project directory: $PROJECT_ROOT"
echo ""

# Activate virtual environment
VENV_PATH="$PROJECT_ROOT/mvenv"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Please run linux_macos.sh again to recreate it."
    exit 1
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment!"
    exit 1
fi

echo "Running Milana UI..."
echo ""
python "$PROJECT_ROOT/ui.py"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Application closed with error code $EXIT_CODE"
fi

# Keep terminal open if double-clicked
if [ "$SHLVL" = 1 ]; then
    echo ""
    read -p "Press Enter to close..." _
fi
EOF

chmod +x "$LAUNCHER"

if [ -f "$LAUNCHER" ]; then
    echo "✓ Launcher created: $LAUNCHER"
else
    echo "✗ ERROR: Failed to create launcher!"
    exit 1
fi
echo ""

# 7) Create desktop entry (only if GUI and xdg-desktop-menu exists)
if [ -n "$DISPLAY" ] && command -v xdg-desktop-menu &> /dev/null; then
    echo "Creating desktop entry..."
    
    DESKTOP_FILE="$HOME/.local/share/applications/Milana.desktop"
    mkdir -p "$(dirname "$DESKTOP_FILE")"
    
    # Check for icon
    ICON_PATH="$PROJECT_ROOT/data/icons/icon.png"
    if [ ! -f "$ICON_PATH" ]; then
        ICON_PATH=""
        echo "Warning: Icon not found at $ICON_PATH"
    fi
    
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=Milana
Comment=Launch Milana Application
Exec=$LAUNCHER
Path=$PROJECT_ROOT
Icon=${ICON_PATH:-$PROJECT_ROOT/data/icons/icon.png}
Terminal=false
Categories=Utility;
EOF
    
    if [ -f "$DESKTOP_FILE" ]; then
        echo "✓ Desktop entry created: $DESKTOP_FILE"
    else
        echo "Warning: Failed to create desktop entry"
    fi
fi

# 8) Final message
echo ""
echo "========================================"
echo "✓ INSTALLATION COMPLETE!"
echo "========================================"
echo ""
echo "To run Milana:"
echo "  $LAUNCHER"
echo ""
echo "Or from project directory:"
echo "  ./run_milana.sh"
echo ""
echo "Project folder: $PROJECT_ROOT"
echo "Virtual environment: $VENV_PATH"
echo "Python version: $PY_VERSION (3.8-3.13)"
echo ""
echo "Troubleshooting:"
echo "  • If run_milana.sh doesn't start: chmod +x run_milana.sh"
echo "  • If Python packages fail: check internet connection"
echo "  • For CUDA support on Linux: pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu118"
echo "  • Python version must be 3.8 - 3.13 (found: $PY_VERSION)"
echo ""

# Keep terminal open if script was double-clicked
if [ "$SHLVL" = 1 ]; then
    echo ""
    read -p "Press Enter to close..." _
fi