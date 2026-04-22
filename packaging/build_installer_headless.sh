#!/usr/bin/env bash
set -euo pipefail

# ========================================
# Kingfisher macOS Builder (Headless)
# Builds unified Kingfisher onedir bundle + .pkg installer
# Called by CI (GitHub Actions) or run locally
# ========================================

echo
printf "%s\n" "========================================"
printf "%s\n" "Kingfisher macOS Builder (Headless)"
printf "%s\n" "========================================"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

cd "${PROJECT_ROOT}"

# Read VERSION.txt from repo root and copy to analyzer folder
if [[ -f "VERSION.txt" ]]; then
  echo "[OK] Reading VERSION.txt from repo root"
  cp "VERSION.txt" "analyzer/VERSION.txt"
  echo "[OK] VERSION.txt copied to analyzer/"
else
  echo "[WARNING] VERSION.txt not found in repo root, generating one..."
  RELEASE_TS="${RELEASE_TS:-$(date "+%Y.%m.%d.%H.%M")}"
  RELEASE_NAME="${RELEASE_NAME:-Kingfisher a${RELEASE_TS}}"
  APP_VERSION="${APP_VERSION:-alpha-${RELEASE_TS}}"
  {
    echo "${APP_VERSION}"
  } > "analyzer/VERSION.txt"
  echo "[OK] Generated VERSION.txt in analyzer/"
fi

# ----------------------------------------
# Activate Python virtual environment
# ----------------------------------------
if [[ -f ".venv2/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv2/bin/activate"
  echo "[OK] Activated .venv2"
else
  echo "[WARNING] .venv2 not found - using system/activated Python"
fi

echo
printf "%s\n" "========================================"
printf "%s\n" "Running PyInstaller (onedir) ..."
printf "%s\n" "========================================"
echo

pushd analyzer || exit 1
python -m PyInstaller Kingfisher-macos.spec
popd

DIST_DIR="analyzer/dist/Kingfisher"
if [[ ! -f "${DIST_DIR}/Kingfisher" ]]; then
  echo "[ERROR] Kingfisher binary not found after build."
  exit 1
fi
echo "[OK] PyInstaller onedir build complete: ${DIST_DIR}/"

echo
printf "%s\n" "========================================"
printf "%s\n" "Build complete!"
printf "%s\n" "========================================"
echo
printf "App Bundle: %s/\n" "${DIST_DIR}"
