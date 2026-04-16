#!/usr/bin/env bash
# === install/build_installer_linux.sh ===
# Creates a .run installer for Milana using makeself

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_installer"
INSTALLER_NAME="Milana_2026.03_Linux.run"

# Check makeself
if ! command -v makeself &>/dev/null; then
    echo "ERROR: makeself not found. Install it first:"
    echo "  sudo apt install makeself   (Debian/Ubuntu)"
    echo "  or from https://github.com/megastep/makeself"
    exit 1
fi

# Clean previous
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Copy the entire project root to BUILD_DIR (excluding development junk)
echo "Copying project files to build directory..."
rsync -av --exclude='mvenv' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.git' --exclude='.idea' --exclude='build' --exclude='dist' \
    --exclude='*.spec' --exclude='install/Output' --exclude='*.iss' \
    "$PROJECT_ROOT/" "$BUILD_DIR/Milana_inst/"

# Create the setup script that will be run after extraction
cat > "$BUILD_DIR/setup.sh" << 'EOF'
#!/bin/bash
# This script runs after files are extracted

INSTALL_DIR="$HOME/.local/share/Milana"
DESKTOP_FILE="$HOME/.local/share/applications/Milana.desktop"
BIN_LINK="$HOME/.local/bin/milana"

echo "========================================"
echo "Milana Installer"
echo "========================================"
echo ""
echo "Default installation directory: $INSTALL_DIR"
read -p "Install here? [Y/n]: " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    INSTALL_DIR="$HOME/.local/share/Milana"
else
    read -p "Enter custom directory: " INSTALL_DIR
fi

mkdir -p "$INSTALL_DIR"
echo "Copying files to $INSTALL_DIR ..."
cp -r "$(dirname "$0")/Milana_inst/"* "$INSTALL_DIR/"

# Make executable
chmod +x "$INSTALL_DIR/Milana"

# Create desktop entry
mkdir -p "$(dirname "$DESKTOP_FILE")"
cat > "$DESKTOP_FILE" << EOL
[Desktop Entry]
Type=Application
Name=Milana
Comment=Milana Application
Exec=$INSTALL_DIR/Milana
Icon=$INSTALL_DIR/data/icons/icon.png
Terminal=false
Categories=Utility;
EOL

# Create symlink in ~/.local/bin
mkdir -p "$HOME/.local/bin"
ln -sf "$INSTALL_DIR/Milana" "$BIN_LINK"

echo ""
echo "Installation complete!"
echo "You can run Milana by:"
echo "  - Desktop shortcut (may appear after logout/login)"
echo "  - Command: milana"
echo "  - Direct: $INSTALL_DIR/Milana"
echo ""
read -p "Press Enter to exit..."
EOF

chmod +x "$BUILD_DIR/setup.sh"

# Create the .run file
echo "Creating .run installer..."
makeself --pigz --threads 0 --follow --target "Milana" \
    "$BUILD_DIR/Milana_inst" \
    "$BUILD_DIR/$INSTALLER_NAME" \
    "Milana Installer" \
    "./setup.sh"

if [ -f "$BUILD_DIR/$INSTALLER_NAME" ]; then
    mv "$BUILD_DIR/$INSTALLER_NAME" "$PROJECT_ROOT/install/"
    echo -e "\033[32m✓ Installer created: $PROJECT_ROOT/install/$INSTALLER_NAME\033[0m"
    echo "To use: run '$PROJECT_ROOT/install/$INSTALLER_NAME' and follow prompts."
else
    echo -e "\033[31m✗ Failed to create installer\033[0m"
    exit 1
fi

rm -rf "$BUILD_DIR"