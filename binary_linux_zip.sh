#!/usr/bin/env bash
# === install/make_linux_zip.sh ===
# Creates a ZIP archive (store mode, no compression) of the Linux build
# Run this script after build_linux.sh

set -e

# Colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Version (match InnoSetup)
VERSION="2026.03"
OUTPUT_NAME="Milana_${VERSION}_Linux.zip"

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_DIR="$(mktemp -d)"

echo "========================================"
echo "Milana Linux ZIP Packager (store mode)"
echo "========================================"

# Check required files
if [ ! -f "$PROJECT_ROOT/Milana" ]; then
    echo -e "${RED}ERROR: $PROJECT_ROOT/Milana not found!${NC}"
    echo "Please run build_linux.sh first."
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/_internal" ]; then
    echo -e "${RED}ERROR: $PROJECT_ROOT/_internal not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found Milana executable and _internal folder${NC}"

# Create temporary package structure
PACKAGE_DIR="$TEMP_DIR/Milana"
mkdir -p "$PACKAGE_DIR"

echo "Copying files to temporary directory..."

# Copy main executable
cp "$PROJECT_ROOT/Milana" "$PACKAGE_DIR/"

# Copy _internal (excluding development junk)
rsync -av --quiet \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.pyd' \
    --exclude='*.spec' \
    --exclude='.git' \
    "$PROJECT_ROOT/_internal/" "$PACKAGE_DIR/_internal/"

# Copy data (exclude user data and db)
rsync -av --quiet \
    --exclude='settings.db' \
    --exclude='chats' \
    --exclude='__pycache__' \
    "$PROJECT_ROOT/data/" "$PACKAGE_DIR/data/"

# Optional: copy a simple run script
cat > "$PACKAGE_DIR/run_milana.sh" << 'EOF'
#!/bin/bash
# Launch Milana from its own directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/Milana"
EOF
chmod +x "$PACKAGE_DIR/run_milana.sh"

# Create README (quick start)
cat > "$PACKAGE_DIR/README.txt" << EOF
Milana version $VERSION

To run Milana:
  - Double-click the "Milana" executable (if your file manager allows)
  - Or run "./run_milana.sh" in a terminal
  - Or run "./Milana" directly

Requirements:
  - glibc 2.31+ (typical on modern Linux distributions)
  - No additional installation required

All files are self-contained in this folder.
EOF

echo -e "${GREEN}✓ Files copied${NC}"

# Create ZIP without compression
echo "Creating ZIP archive (store mode)..."
cd "$TEMP_DIR"
zip -r -0 "$OUTPUT_NAME" "Milana/" > /dev/null

# Move to project root (or install folder)
if [ ! -d "$PROJECT_ROOT/install" ]; then
    mkdir -p "$PROJECT_ROOT/install"
fi
mv "$OUTPUT_NAME" "$PROJECT_ROOT/install/"

# Cleanup
cd "$PROJECT_ROOT"
rm -rf "$TEMP_DIR"

echo -e "${GREEN}✓ Archive created: install/$OUTPUT_NAME${NC}"
ls -lh "$PROJECT_ROOT/install/$OUTPUT_NAME"

echo ""
echo "========================================"
echo -e "${GREEN}Done!${NC}"
echo "Archive is located at: $PROJECT_ROOT/install/$OUTPUT_NAME"
echo "It contains: Milana, _internal, data, run_milana.sh, README.txt"
echo "========================================"