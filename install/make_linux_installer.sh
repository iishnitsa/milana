#!/usr/bin/env bash
# Создаёт самораспаковывающийся .run установщик на основе tar.gz
# Работает быстро, так как сжатие происходит только один раз (tar+gz)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_installer"
INSTALLER_NAME="Milana_2026.03_Linux.run"

# Проверяем, что сборка приложения выполнена
if [ ! -f "$PROJECT_ROOT/Milana" ] || [ ! -d "$PROJECT_ROOT/_internal" ]; then
    echo "ERROR: Run ./install/build_linux.sh first to create Milana and _internal"
    exit 1
fi

# Очистка временной папки
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/package"

# ---- Копируем только нужные файлы с исключениями (как в InnoSetup) ----
echo "Copying application files (with exclusions)..."
rsync -av --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.pyd' \
    --exclude='*.db' \
    --exclude='chats/*' \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='.git*' \
    --exclude='.vscode' \
    --exclude='.idea' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='*.spec' \
    --exclude='mvenv' \
    --exclude='install/Output' \
    --exclude='Output' \
    --exclude='*.iss' \
    --exclude='*.run' \
    --exclude='build_installer' \
    "$PROJECT_ROOT/Milana" \
    "$PROJECT_ROOT/_internal" \
    "$PROJECT_ROOT/data" \
    "$BUILD_DIR/package/"

# ---- Создаём установочный скрипт, который будет добавлен в начало .run ----
cat > "$BUILD_DIR/installer.sh" << 'EOF'
#!/bin/bash
# Самораспаковывающийся установщик Milana

set -e

# Находим строку, где начинается архив
ARCHIVE_LINE=$(awk '/^__ARCHIVE_BELOW__/ {print NR + 1; exit 0;}' "$0")
INSTALL_DIR="$HOME/.local/share/Milana"
DESKTOP_FILE="$HOME/.local/share/applications/Milana.desktop"
BIN_LINK="$HOME/.local/bin/milana"

echo "========================================"
echo "Milana Installer for Linux"
echo "========================================"
echo "Default installation: $INSTALL_DIR"
read -p "Install here? [Y/n]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    INSTALL_DIR="$HOME/.local/share/Milana"
else
    read -p "Enter custom path: " INSTALL_DIR
fi

# Создаём временную папку для распаковки
TMP_DIR=$(mktemp -d)
echo "Extracting files..."
tail -n +$ARCHIVE_LINE "$0" | tar xz -C "$TMP_DIR"

# Копируем в целевую директорию
mkdir -p "$INSTALL_DIR"
cp -r "$TMP_DIR/package/"* "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/Milana"

# Desktop entry
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

# Символическая ссылка в ~/.local/bin
mkdir -p "$HOME/.local/bin"
ln -sf "$INSTALL_DIR/Milana" "$BIN_LINK"

# Очистка
rm -rf "$TMP_DIR"

echo ""
echo "✓ Installation complete!"
echo "  Run: milana"
echo "  or: $INSTALL_DIR/Milana"
echo "  Desktop shortcut added (may need logout/login)"
echo ""
read -p "Press Enter to exit..."
exit 0

__ARCHIVE_BELOW__
EOF

# ---- Создаём tar.gz архив из package ----
echo "Creating tar.gz archive (this is fast)..."
tar czf "$BUILD_DIR/package.tar.gz" -C "$BUILD_DIR" package

# ---- Объединяем installer.sh и архив в один .run файл ----
echo "Building .run installer..."
cat "$BUILD_DIR/installer.sh" "$BUILD_DIR/package.tar.gz" > "$BUILD_DIR/$INSTALLER_NAME"
chmod +x "$BUILD_DIR/$INSTALLER_NAME"

# ---- Перемещаем готовый установщик в папку install/ ----
mv "$BUILD_DIR/$INSTALLER_NAME" "$PROJECT_ROOT/install/"

# ---- Очистка ----
rm -rf "$BUILD_DIR"

echo ""
echo "✓ Installer created: $PROJECT_ROOT/install/$INSTALLER_NAME"
echo "  Size: $(du -h "$PROJECT_ROOT/install/$INSTALLER_NAME" | cut -f1)"
echo ""
echo "To test:"
echo "  $PROJECT_ROOT/install/$INSTALLER_NAME"