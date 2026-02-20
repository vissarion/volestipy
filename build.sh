#!/usr/bin/env bash
# build.sh - convenience script to build volestipy from source
#
# Usage:
#   ./build.sh                          # auto-detect volesti in external/volesti
#   VOLESTI_INCLUDE_DIR=/path ./build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
VOLESTI_INCLUDE_DIR="${VOLESTI_INCLUDE_DIR:-${SCRIPT_DIR}/external/volesti/include}"

echo "======================================================"
echo " volestipy build script"
echo "======================================================"
echo " Source:          ${SCRIPT_DIR}"
echo " Build dir:       ${BUILD_DIR}"
echo " volesti include: ${VOLESTI_INCLUDE_DIR}"
echo "======================================================"

# ── Check prerequisites ───────────────────────────────────────────────────────
for cmd in cmake python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' is required but not found on PATH." >&2
        exit 1
    fi
done

# ── Check volesti headers ─────────────────────────────────────────────────────
if [ ! -f "${VOLESTI_INCLUDE_DIR}/cartesian_geom/cartesian_kernel.h" ]; then
    echo ""
    echo "ERROR: volesti headers not found at '${VOLESTI_INCLUDE_DIR}'."
    echo ""
    echo "Options:"
    echo "  1. Run:  git submodule update --init --recursive"
    echo "  2. Set:  VOLESTI_INCLUDE_DIR=/path/to/volesti/include ./build.sh"
    exit 1
fi

# ── Configure ─────────────────────────────────────────────────────────────────
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVOLESTI_INCLUDE_DIR="${VOLESTI_INCLUDE_DIR}" \
    -DDISABLE_LPSOLVE=OFF \
    -DPYTHON_EXECUTABLE="$(command -v python3)"

# ── Build ─────────────────────────────────────────────────────────────────────
cmake --build . --config Release -- -j"$(nproc 2>/dev/null || echo 4)"

# ── Copy the extension next to the Python package for in-place use ────────────
SO_FILE="$(find . -name '_volestipy*.so' -o -name '_volestipy*.pyd' 2>/dev/null | head -1)"
if [ -n "${SO_FILE}" ]; then
    cp "${SO_FILE}" "${SCRIPT_DIR}/volestipy/"
    echo ""
    echo "Extension copied to ${SCRIPT_DIR}/volestipy/"
fi

echo ""
echo "======================================================"
echo " Build complete."
echo " To use:  PYTHONPATH=${SCRIPT_DIR} python3"
echo " Or run:  pip install -e ${SCRIPT_DIR}"
echo "======================================================"
