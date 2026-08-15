#!/usr/bin/env bash
# Builds a self-extracting .run installer for Milana (like InnoSetup on Windows)
# Usage: ./make_linux_installer.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_installer"
INSTALLER_NAME="MilanaSetup.run"

# Verify that the application has been built
if [ ! -f "$PROJECT_ROOT/Milana" ] || [ ! -d "$PROJECT_ROOT/_internal" ]; then
    echo "ERROR: Run ./install/build_linux.sh first to create Milana and _internal"
    exit 1
fi

# Clean temporary build folder
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/package"

# Copy entire project root with exclusions (same as InnoSetup)
echo "Copying application files (with exclusions)..."
rsync -av --delete \
    --exclude='__pycache__' \
    --exclude='__pycache__/*' \
    --exclude='tests' \
    --exclude='install/Output' \
    --exclude='install/Output/*' \
    --exclude='mvenv' \
    --exclude='mvenv/*' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.pyd' \
    --exclude='launcher.py' \
    --exclude='data/settings.db' \
    --exclude='run_ui.cmd' \
    --exclude='run_ui.sh' \
    --exclude='*.lnk' \
    --exclude='build' \
    --exclude='build/*' \
    --exclude='dist' \
    --exclude='dist/*' \
    --exclude='Milana.lnk' \
    --exclude='Output' \
    --exclude='Output/*' \
    --exclude='.git' \
    --exclude='.git/*' \
    --exclude='.gitattributes' \
    --exclude='.vscode' \
    --exclude='.vscode/*' \
    --exclude='.idea' \
    --exclude='.idea/*' \
    --exclude='*.log' \
    --exclude='*.bak' \
    --exclude='*.tmp' \
    --exclude='thumbs.db' \
    --exclude='*.db' \
    --exclude='data/chats' \
    --exclude='launch_milana.cmd' \
    --exclude='run_milana.sh' \
    --exclude='*.run' \
    --exclude='build_installer' \
    --exclude='buildexe.bat' \
    --exclude='build_linux.sh' \
    --exclude='make_linux_installer.sh' \
    --exclude='*.iss' \
    --exclude='requirements.txt' \
    "$PROJECT_ROOT/" "$BUILD_DIR/package/"

# Create the installer script (self-extracting archive with GUI)
cat > "$BUILD_DIR/installer.sh" << 'EOF'
#!/bin/bash
# Milana Self-Extracting Installer (graphical + console fallback)

set -e

# ----------------------------------------------------------------------
# Detect available GUI tool
# ----------------------------------------------------------------------
if command -v zenity &> /dev/null; then
    USE_GUI="zenity"
else
    USE_GUI="none"
fi

# ----------------------------------------------------------------------
# Get standard XDG directories
# ----------------------------------------------------------------------
get_desktop_dir() {
    if [ -n "$XDG_DESKTOP_DIR" ]; then
        echo "$XDG_DESKTOP_DIR"
    elif [ -d "$HOME/Desktop" ]; then
        echo "$HOME/Desktop"
    elif [ -d "$HOME/Рабочий стол" ]; then
        echo "$HOME/Рабочий стол"
    elif [ -d "$HOME/桌面" ]; then
        echo "$HOME/桌面"
    else
        echo "$HOME/Desktop"  # Default, will be created if needed
    fi
}

get_applications_dir() {
    if [ -n "$XDG_DATA_HOME" ]; then
        echo "$XDG_DATA_HOME/applications"
    else
        echo "$HOME/.local/share/applications"
    fi
}

get_bin_dir() {
    if [ -n "$XDG_BIN_HOME" ]; then
        echo "$XDG_BIN_HOME"
    else
        echo "$HOME/.local/bin"
    fi
}

# ----------------------------------------------------------------------
# Universal wrappers for GUI and console
# ----------------------------------------------------------------------
show_info() {
    local msg="$1"
    if [ "$USE_GUI" = "zenity" ]; then
        zenity --info --title="Milana Setup" --text="$msg" --width=450
    else
        echo -e "\n$msg"
        read -p "Press Enter to continue..."
    fi
}

show_error() {
    local msg="$1"
    if [ "$USE_GUI" = "zenity" ]; then
        zenity --error --title="Milana Setup Error" --text="$msg" --width=450
    else
        echo -e "\nERROR: $msg" >&2
        read -p "Press Enter to continue..."
    fi
}

ask_yesno() {
    local msg="$1"
    if [ "$USE_GUI" = "zenity" ]; then
        zenity --question --title="Milana Setup" --text="$msg" --width=450
    else
        local ans
        read -p "$msg (y/N): " ans
        [[ "$ans" =~ [Yy] ]]
    fi
}

# Directory selection (parent folder, then we append /Milana)
ask_parent_directory() {
    local default_parent="/opt"
    if [ -w "/opt" ]; then
        default_parent="/opt"
    else
        default_parent="$HOME/.local"
    fi
    
    if [ "$USE_GUI" = "zenity" ]; then
        local parent=$(zenity --file-selection --directory --title="Choose installation folder (Milana will be placed inside)" --filename="$default_parent")
        if [ $? -eq 0 ] && [ -n "$parent" ]; then
            echo "$parent"
        else
            echo "$default_parent"
        fi
    else
        local val
        echo ""
        echo "Choose installation parent directory:"
        echo "  /opt        - System-wide (may require sudo)"
        echo "  ~/.local    - User-only"
        echo ""
        read -p "Parent directory [$default_parent]: " val
        echo "${val:-$default_parent}"
    fi
}

# Ask about shortcuts with proper desktop path detection
ask_shortcuts() {
    local desktop_dir=$(get_desktop_dir)
    local menu_dir=$(get_applications_dir)
    
    if [ "$USE_GUI" = "zenity" ]; then
        zenity --list --checklist \
            --title="Milana Setup" \
            --text="Choose shortcuts to create:\n\nDesktop folder: $desktop_dir\nApplication menu: $menu_dir" \
            --column="Create" --column="Shortcut" --column="Location" \
            FALSE "Desktop shortcut" "$desktop_dir" \
            FALSE "Application menu shortcut" "$menu_dir" \
            --width=550 --height=250
    else
        echo ""
        echo "Choose shortcuts to create:"
        echo "  Desktop folder: $desktop_dir"
        echo "  Application menu: $menu_dir"
        echo ""
        
        local create_desktop=false
        local create_menu=false
        
        read -p "Create Desktop shortcut? (y/N): " ans
        [[ "$ans" =~ [Yy] ]] && create_desktop=true
        
        read -p "Create Application Menu shortcut? (y/N): " ans
        [[ "$ans" =~ [Yy] ]] && create_menu=true
        
        if [ "$create_desktop" = true ] && [ "$create_menu" = true ]; then
            echo "both"
        elif [ "$create_desktop" = true ]; then
            echo "desktop"
        elif [ "$create_menu" = true ]; then
            echo "menu"
        else
            echo "none"
        fi
    fi
}

show_progress() {
    local msg="$1"
    if [ "$USE_GUI" = "zenity" ]; then
        echo "10" | zenity --progress --title="Milana Setup" --text="$msg" --width=450 --auto-close --no-cancel --percentage=10 --auto-kill
    else
        echo "$msg..."
    fi
}

# ----------------------------------------------------------------------
# Main installation routine
# ----------------------------------------------------------------------
main_installation() {
    # Welcome
    show_info "Welcome to Milana Setup Wizard\n\nThis will install Milana on your system."
    
    # Choose parent directory
    PARENT_DIR=$(ask_parent_directory)
    INSTALL_DIR="$PARENT_DIR/Milana"
    
    # Check if we can write to parent directory
    if [ ! -w "$PARENT_DIR" ]; then
        if ! ask_yesno "You don't have write permission to '$PARENT_DIR'.\n\nDo you want to try installing with sudo?"; then
            show_info "Installation cancelled."
            exit 0
        fi
        USE_SUDO=true
    else
        USE_SUDO=false
    fi
    
    # Ask about shortcuts
    SHORTCUTS=$(ask_shortcuts)

    # Optional image models (BLIP + EasyOCR, ~1 GB)
    INSTALL_MODELS=false
    if ask_yesno "Install image recognition models?\n\nOCR / image captions (BLIP + EasyOCR).\nAdds about 1 GB on disk.\n\nWithout models, image recognition is disabled."; then
        INSTALL_MODELS=true
    fi
    
    # Confirmation
    local confirm_msg="Ready to install?\n\nDestination folder: $INSTALL_DIR"
    if [ "$INSTALL_MODELS" = true ]; then
        confirm_msg="$confirm_msg\nImage models: yes"
    else
        confirm_msg="$confirm_msg\nImage models: no (OCR off)"
    fi
    
    if [ "$USE_SUDO" = true ]; then
        confirm_msg="$confirm_msg\n(Will use sudo for installation)"
    fi
    
    if ! ask_yesno "$confirm_msg"; then
        show_info "Installation cancelled."
        exit 0
    fi
    
    # Extract the embedded archive
    show_progress "Extracting files..."
    ARCHIVE_LINE=$(awk '/^__ARCHIVE_BELOW__/ {print NR + 1; exit 0;}' "$0")
    TMP_DIR=$(mktemp -d)
    tail -n +$ARCHIVE_LINE "$0" | tar xz -C "$TMP_DIR"
    
    # Verify extraction
    if [ ! -d "$TMP_DIR/package" ]; then
        show_error "Failed to extract installation files."
        rm -rf "$TMP_DIR"
        exit 1
    fi

    # Drop models from package if user declined (installer still contains them; strip before copy)
    if [ "$INSTALL_MODELS" != true ]; then
        show_progress "Skipping image models..."
        rm -rf "$TMP_DIR/package/data/models" 2>/dev/null || true
        mkdir -p "$TMP_DIR/package/data/models"
        # marker for support / docs
        echo "skipped" > "$TMP_DIR/package/data/models/.models_not_installed" 2>/dev/null || true
    fi
    
    # Copy to destination
    show_progress "Installing files to $INSTALL_DIR..."
    
    if [ "$USE_SUDO" = true ]; then
        sudo mkdir -p "$INSTALL_DIR"
        sudo cp -r "$TMP_DIR/package/"* "$INSTALL_DIR/"
        sudo chmod +x "$INSTALL_DIR/Milana"
        # Fix ownership
        if [ "$USE_SUDO" = true ] && [ -n "$SUDO_USER" ]; then
            sudo chown -R "$SUDO_USER":"$SUDO_USER" "$INSTALL_DIR" 2>/dev/null || true
        fi
    else
        mkdir -p "$INSTALL_DIR"
        cp -r "$TMP_DIR/package/"* "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/Milana"
    fi
    
    # Verify installation
    if [ ! -f "$INSTALL_DIR/Milana" ]; then
        show_error "Installation failed: Milana binary not found in $INSTALL_DIR"
        rm -rf "$TMP_DIR"
        exit 1
    fi
    
    # Create shortcuts based on user choice
    local desktop_dir=$(get_desktop_dir)
    local menu_dir=$(get_applications_dir)
    
    # Create Application menu shortcut
    if [[ "$SHORTCUTS" == *"menu"* ]] || [[ "$SHORTCUTS" == "both" ]] || [ "$SHORTCUTS" = "menu" ]; then
        show_progress "Creating application menu shortcut..."
        mkdir -p "$menu_dir"
        local desktop_file="$menu_dir/Milana.desktop"
        
        cat > "$desktop_file" << EOL
[Desktop Entry]
Type=Application
Name=Milana
Comment=Milana AI Assistant
Exec=$INSTALL_DIR/Milana
Icon=$INSTALL_DIR/data/icons/icon.png
Terminal=false
Categories=Utility;Office;
StartupWMClass=Milana
EOL
        chmod +x "$desktop_file"
        
        # Update desktop database if available
        if command -v update-desktop-database &> /dev/null; then
            update-desktop-database "$menu_dir" 2>/dev/null || true
        fi
    fi
    
    # Create Desktop shortcut
    if [[ "$SHORTCUTS" == *"desktop"* ]] || [[ "$SHORTCUTS" == "both" ]] || [ "$SHORTCUTS" = "desktop" ]; then
        show_progress "Creating desktop shortcut..."
        mkdir -p "$desktop_dir"
        local desktop_shortcut="$desktop_dir/Milana.desktop"
        
        cat > "$desktop_shortcut" << EOL
[Desktop Entry]
Type=Application
Name=Milana
Comment=Milana AI Assistant
Exec=$INSTALL_DIR/Milana
Icon=$INSTALL_DIR/data/icons/icon.png
Terminal=false
Categories=Utility;Office;
EOL
        chmod +x "$desktop_shortcut"
    fi
    
    # Create symbolic link in ~/.local/bin
    show_progress "Creating command-line launcher..."
    local bin_dir=$(get_bin_dir)
    mkdir -p "$bin_dir"
    ln -sf "$INSTALL_DIR/Milana" "$bin_dir/milana"
    
    # Add to PATH if needed
    if [[ ":$PATH:" != *":$bin_dir:"* ]]; then
        echo ""
        echo "NOTE: $bin_dir is not in your PATH."
        echo "Add this line to your ~/.bashrc or ~/.zshrc:"
        echo "  export PATH=\"\$PATH:$bin_dir\""
    fi
    
    # Cleanup
    rm -rf "$TMP_DIR"
    
    # Final message with verification
    local final_msg="Installation complete!\n\n"
    final_msg="$final_msg✓ Installed to: $INSTALL_DIR\n"
    final_msg="$final_msg✓ Files copied: $(find "$INSTALL_DIR" -type f | wc -l) files\n"
    if [ "$INSTALL_MODELS" = true ]; then
        final_msg="$final_msg✓ Image models: installed\n"
    else
        final_msg="$final_msg✓ Image models: skipped (OCR disabled)\n"
    fi
    final_msg="$final_msg✓ Command-line launcher: milana\n"
    
    if [[ "$SHORTCUTS" == *"menu"* ]] || [[ "$SHORTCUTS" == "both" ]]; then
        final_msg="$final_msg✓ Application menu: Milana\n"
    fi
    if [[ "$SHORTCUTS" == *"desktop"* ]] || [[ "$SHORTCUTS" == "both" ]]; then
        final_msg="$final_msg✓ Desktop shortcut: $desktop_dir/Milana.desktop\n"
    fi
    
    final_msg="$final_msg\nYou can run Milana by typing 'milana' in terminal."
    
    show_info "$final_msg"
    exit 0
}

# Run with error handling
if main_installation; then
    exit 0
else
    show_error "An error occurred during installation."
    exit 1
fi

__ARCHIVE_BELOW__
EOF

# Create tar.gz archive of the package folder
echo "Creating tar.gz archive..."
tar czf "$BUILD_DIR/package.tar.gz" -C "$BUILD_DIR" package

# Concatenate installer script and archive into a single .run file
echo "Building .run installer..."
cat "$BUILD_DIR/installer.sh" "$BUILD_DIR/package.tar.gz" > "$BUILD_DIR/$INSTALLER_NAME"
chmod +x "$BUILD_DIR/$INSTALLER_NAME"

# Move final installer to the install/ folder
mv "$BUILD_DIR/$INSTALLER_NAME" "$PROJECT_ROOT/install/"

# Clean up build directory
rm -rf "$BUILD_DIR"

echo ""
echo "✓ Installer created: $PROJECT_ROOT/install/$INSTALLER_NAME"
echo "  Size: $(du -h "$PROJECT_ROOT/install/$INSTALLER_NAME" | cut -f1)"
echo ""
echo "To test:"
echo "  $PROJECT_ROOT/install/$INSTALLER_NAME"