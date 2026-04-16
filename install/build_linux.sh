#!/usr/bin/env bash
# === install/build_linux.sh ===
# Build Milana as onedir executable for Linux

set -e

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "Milana Linux Build Script (onedir)"
echo "========================================"

# 1) Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/mvenv"
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"
INTERNAL_DIR="$PROJECT_ROOT/_internal"
EXE_FILE="$PROJECT_ROOT/Milana"

echo "Project root: $PROJECT_ROOT"

# 2) Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -rf "$BUILD_DIR" "$DIST_DIR" "$INTERNAL_DIR" "$EXE_FILE" "$PROJECT_ROOT/Milana.spec"

# 3) Check virtual environment
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo -e "${RED}ERROR: Virtual environment not found at $VENV_DIR${NC}"
    echo "Please run ../linux_macos.sh first to create venv."
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment found${NC}"

# 4) Activate venv
source "$VENV_DIR/bin/activate"

# 5) Ensure pyinstaller is installed
if ! pip show pyinstaller &>/dev/null; then
    echo "Installing pyinstaller..."
    pip install pyinstaller
fi

# 6) Check launcher.py exists
if [ ! -f "$PROJECT_ROOT/launcher.py" ]; then
    echo -e "${RED}ERROR: launcher.py not found in project root!${NC}"
    exit 1
fi

# 7) Run PyInstaller
echo -e "${YELLOW}Running PyInstaller...${NC}"
pyinstaller --onedir \
    --name "Milana" \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR" \
    --add-data "$PROJECT_ROOT/data:data" \
    --collect-data ddgs \
    --collect-all fake_useragent \
    --collect-all easyocr \
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
    --hidden-import=PIL \
    --hidden-import=PIL._tkinter_finder \
    --hidden-import=tkinter \
    --noconsole \
    --clean \
    --noconfirm \
    "$PROJECT_ROOT/launcher.py"

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: PyInstaller build failed!${NC}"
    exit 1
fi

# 8) Move built files to project root
echo -e "${YELLOW}Moving files to project root...${NC}"
if [ -d "$DIST_DIR/Milana" ]; then
    # Copy all contents of dist/Milana/ to project root
    cp -r "$DIST_DIR/Milana/"* "$PROJECT_ROOT/"
    # Rename the main folder Milana -> _internal
    if [ -d "$PROJECT_ROOT/Milana" ]; then
        mv "$PROJECT_ROOT/Milana" "$PROJECT_ROOT/_internal"
        echo -e "${GREEN}✓ Renamed Milana folder to _internal${NC}"
    fi
else
    echo -e "${RED}ERROR: $DIST_DIR/Milana not found${NC}"
    exit 1
fi

# 9) Clean temporary files
echo -e "${YELLOW}Cleaning temporary files...${NC}"
rm -rf "$BUILD_DIR" "$DIST_DIR" "$PROJECT_ROOT/Milana.spec"

# 10) Verify
if [ -f "$PROJECT_ROOT/Milana" ]; then
    echo -e "${GREEN}✓ Executable: $PROJECT_ROOT/Milana${NC}"
    ls -lh "$PROJECT_ROOT/Milana"
else
    echo -e "${RED}✗ ERROR: Milana executable not found!${NC}"
    exit 1
fi

if [ -d "$PROJECT_ROOT/_internal" ]; then
    echo -e "${GREEN}✓ _internal folder exists${NC}"
else
    echo -e "${RED}✗ ERROR: _internal folder missing${NC}"
    exit 1
fi

echo ""
echo "========================================"
echo -e "${GREEN}Build completed successfully!${NC}"
echo "========================================"
echo ""
echo "To run Milana:"
echo "  $PROJECT_ROOT/Milana"
echo ""