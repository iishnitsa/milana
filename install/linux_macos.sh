#!/usr/bin/env bash
# === linux_macos.sh (pyenv version) ===
# Uses pyenv to install Python 3.13.7 if missing, then creates venv

set -e

# 1) Determine project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# 2) Check for pyenv
if ! command -v pyenv &> /dev/null; then
    echo "ERROR: pyenv is not installed."
    echo "Please install pyenv first, then re-run this script."
    exit 1
fi

echo "✓ pyenv found: $(which pyenv)"

# 3) Install Python 3.13.7 via pyenv if not already present
PYTHON_VERSION="3.13.7"
echo "Checking for Python $PYTHON_VERSION via pyenv..."

if pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
    echo "Python $PYTHON_VERSION already installed."
else
    echo "Python $PYTHON_VERSION not found. Installing..."
    pyenv install "$PYTHON_VERSION"
fi

# Get the full path to the interpreter
PYENV_ROOT="$(pyenv root)"
PYTHON_CMD="$PYENV_ROOT/versions/$PYTHON_VERSION/bin/python3.13"

if [ ! -f "$PYTHON_CMD" ]; then
    echo "ERROR: Cannot find Python $PYTHON_VERSION at $PYTHON_CMD"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION found: $PYTHON_CMD"

# 4) Create/verify virtual environment using this specific Python
VENV_PATH="$PROJECT_ROOT/mvenv"

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists at $VENV_PATH"
    printf "Recreate? (y/N): "
    read -r yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        echo "Removing old virtual environment..."
        rm -rf "$VENV_PATH"
        echo "Creating new virtual environment with Python $PYTHON_VERSION..."
        "$PYTHON_CMD" -m venv "$VENV_PATH"
    else
        echo "Using existing virtual environment."
    fi
else
    echo "Creating virtual environment with Python $PYTHON_VERSION..."
    "$PYTHON_CMD" -m venv "$VENV_PATH"
fi

# Verify virtual environment
if [ ! -f "$VENV_PATH/bin/python" ]; then
    echo "ERROR: Virtual environment creation failed!"
    exit 1
fi
echo "✓ Virtual environment verified."
echo ""

# 5) Activate venv and install dependencies
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

# 8) Create desktop entry (only if GUI and xdg-desktop-menu exists)
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

# 9) Final message
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
echo "Python version: $PYTHON_VERSION"
echo ""
echo "Troubleshooting:"
echo "  • If run_milana.sh doesn't start: chmod +x run_milana.sh"
echo "  • If Python packages fail: check internet connection"
echo "  • For CUDA support on Linux: pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu118"
echo "  • Python version is fixed to $PYTHON_VERSION via pyenv"
echo ""

# Keep terminal open if script was double-clicked
if [ "$SHLVL" = 1 ]; then
    echo ""
    read -p "Press Enter to close..." _
fi
