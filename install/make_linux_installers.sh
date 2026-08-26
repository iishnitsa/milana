#!/usr/bin/env bash
# Build both Linux .run installers:
#   MilanaSetup.run              (models packed; asked at install)
#   MilanaSetupNoOCRModels.run   (data/models not captured)
# Usage: ./make_linux_installers.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$PROJECT_ROOT/Milana" ] || [ ! -d "$PROJECT_ROOT/_internal" ]; then
    echo "ERROR: Run ./install/buildlinux.sh first to create Milana and _internal"
    exit 1
fi

echo "========================================"
echo "MilanaSetup.run"
echo "========================================"
"$SCRIPT_DIR/make_linux_installer.sh"

echo ""
echo "========================================"
echo "MilanaSetupNoOCRModels.run"
echo "========================================"
"$SCRIPT_DIR/make_linux_installer_no_ocr_models.sh"

echo ""
echo "Both installers written to:"
echo "  $PROJECT_ROOT/install/MilanaSetup.run"
echo "  $PROJECT_ROOT/install/MilanaSetupNoOCRModels.run"
