#!/usr/bin/env bash
# === linux_macos.sh ===

set -e

# 1) Determine project root (one level up from install/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 2) Create Python virtual environment 'mvenv'
python3 -m venv "$PROJECT_ROOT/mvenv"

# 3) Activate venv and install dependencies
# shellcheck source=/dev/null
source "$PROJECT_ROOT/mvenv/bin/activate"
pip install --upgrade pip
pip install -r "$PROJECT_ROOT/install/requirements.txt"

# 4) Write launcher script run_ui.sh in project root
cat > "$PROJECT_ROOT/run_ui.sh" << 'EOF'
#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$PROJECT_ROOT/mvenv/bin/activate"
python "$PROJECT_ROOT/ui.py"
EOF
chmod +x "$PROJECT_ROOT/run_ui.sh"
echo "Created launcher: $PROJECT_ROOT/run_ui.sh"

# 5) Optionally create a .desktop file for Linux
if command -v xdg-desktop-icon >/dev/null 2>&1; then
  DESKTOP_FILE="$HOME/Desktop/Milana.desktop"
  cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=Milana
Exec=$PROJECT_ROOT/run_ui.sh
Icon=$PROJECT_ROOT/data/icons/icon.png
Terminal=false
EOF
  chmod +x "$DESKTOP_FILE"
  echo "Created desktop entry: $DESKTOP_FILE"
fi

echo "Installation complete!"
