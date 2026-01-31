#!/usr/bin/env bash
# === linux_macos.sh ===

set -e

# 1) Определяем корень проекта
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# 2) Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python not found! Install Python 3.8+ first."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Using Python: $(which $PYTHON_CMD)"
echo "Python version: $($PYTHON_CMD --version)"

# 3) Создаем виртуальное окружение (если еще нет)
VENV_PATH="$PROJECT_ROOT/mvenv"
if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists at $VENV_PATH"
    read -p "Recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old virtual environment..."
        rm -rf "$VENV_PATH"
        $PYTHON_CMD -m venv "$VENV_PATH"
    fi
else
    echo "Creating virtual environment at $VENV_PATH..."
    $PYTHON_CMD -m venv "$VENV_PATH"
fi

# 4) Активируем venv и устанавливаем зависимости
source "$VENV_PATH/bin/activate"
echo "Virtual environment activated"
echo "Python in venv: $(which python)"

pip install --upgrade pip
echo "Installing dependencies from requirements.txt..."
pip install -r "$PROJECT_ROOT/install/requirements.txt"

# 5) Создаем скрипт запуска
cat > "$PROJECT_ROOT/run_ui.sh" << 'EOF'
#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_ROOT/mvenv/bin/activate"
python "$PROJECT_ROOT/ui.py"
EOF

chmod +x "$PROJECT_ROOT/run_ui.sh"
echo "Created launcher: $PROJECT_ROOT/run_ui.sh"

# 6) Создаем .desktop файл только если есть GUI
if [ -n "$DISPLAY" ] && command -v xdg-desktop-menu >/dev/null 2>&1; then
    DESKTOP_FILE="$HOME/.local/share/applications/Milana.desktop"
    
    # Проверяем наличие иконки
    ICON_PATH="$PROJECT_ROOT/data/icons/icon.png"
    if [ ! -f "$ICON_PATH" ]; then
        echo "Warning: Icon not found at $ICON_PATH"
        ICON_PATH=""
    fi
    
    mkdir -p "$(dirname "$DESKTOP_FILE")"
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=Milana
Comment=Launch Milana Application
Exec=$PROJECT_ROOT/run_ui.sh
Path=$PROJECT_ROOT
Icon=${ICON_PATH}
Terminal=false
Categories=Utility;
EOF
    
    echo "Created desktop entry: $DESKTOP_FILE"
fi

echo ""
echo "========================================"
echo "Installation complete!"
echo "To run Milana, use:"
echo "  $PROJECT_ROOT/run_ui.sh"
echo "Or from project directory: ./run_ui.sh"
echo "========================================"