#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Create Python wheel for ptoas.
#
# Usage: ./create_wheel.sh
#
# Required environment variables:
#   PTO_SOURCE_DIR  - Path to PTO source directory
#   PTO_INSTALL_DIR - Path to PTO install directory
#   LLVM_BUILD_DIR  - Path to LLVM build directory (for python packages location)
#
# Optional environment variables:
#   WHEEL_PLAT_NAME - Explicit wheel platform tag (for bdist_wheel --plat-name)
#   PTO_WHEEL_STAGING_DIR - Temporary wheel staging directory
#   PTO_WHEEL_DIST_DIR - Directory for built wheel artifacts
#   PTOAS_PYTHON_PACKAGE_VERSION - Wheel package version override
#   PYTHON - Python interpreter to use for wheel metadata and setup.py

set -e

# Validate required environment variables
for var in PTO_SOURCE_DIR PTO_INSTALL_DIR LLVM_BUILD_DIR; do
  if [ -z "${!var}" ]; then
    echo "Error: $var environment variable is not set" >&2
    exit 1
  fi
done

_find_mlir_python_package() {
  local prefix="$1"
  local cand
  for cand in \
    "${prefix}/tools/mlir/python_packages/mlir_core" \
    "${prefix}/python_packages/mlir_core"
  do
    if [ -d "$cand" ]; then
      echo "$cand"
      return 0
    fi
  done
  echo "Error: cannot find MLIR python package under ${prefix}" >&2
  echo "Tried:" >&2
  echo "  ${prefix}/tools/mlir/python_packages/mlir_core" >&2
  echo "  ${prefix}/python_packages/mlir_core" >&2
  exit 1
}
MLIR_PYTHON_PACKAGE_DIR="$(_find_mlir_python_package "${LLVM_BUILD_DIR}")"
WHEEL_STAGING_DIR="${PTO_WHEEL_STAGING_DIR:-${PTO_SOURCE_DIR}/build/wheel-staging}"
WHEEL_DIST_DIR="${PTO_WHEEL_DIST_DIR:-${PTO_SOURCE_DIR}/build/wheel-dist}"
PYTHON_BIN="${PYTHON:-python3}"
PTOAS_PYTHON_PACKAGE_VERSION="${PTOAS_PYTHON_PACKAGE_VERSION:-${PTOAS_VERSION:-}}"
if [ -z "${PTOAS_PYTHON_PACKAGE_VERSION}" ]; then
  PTOAS_PYTHON_PACKAGE_VERSION="$("${PYTHON_BIN}" "${PTO_SOURCE_DIR}/.github/scripts/compute_ptoas_version.py" --cmake-file "${PTO_SOURCE_DIR}/CMakeLists.txt" --mode dev)"
fi
export PTOAS_PYTHON_PACKAGE_VERSION

echo "Creating Python wheel..."
echo "Wheel package version: ${PTOAS_PYTHON_PACKAGE_VERSION}"

rm -rf "${WHEEL_STAGING_DIR}" "${WHEEL_DIST_DIR}"
mkdir -p "${WHEEL_STAGING_DIR}" "${WHEEL_DIST_DIR}"

# Build the wheel from an isolated staging tree. The LLVM MLIR Python package is
# a read-only input; PTOAS files are overlaid only inside WHEEL_STAGING_DIR.
echo "Copying MLIR Python package into wheel staging..."
cp -a "${MLIR_PYTHON_PACKAGE_DIR}/." "${WHEEL_STAGING_DIR}/"

echo "Overlaying PTO dialect files..."
mkdir -p "${WHEEL_STAGING_DIR}/mlir/dialects"
cp "${PTO_INSTALL_DIR}/mlir/dialects/"*.py "${WHEEL_STAGING_DIR}/mlir/dialects/"

echo "Overlaying PTO native extension..."
mkdir -p "${WHEEL_STAGING_DIR}/mlir/_mlir_libs"
cp "${PTO_INSTALL_DIR}/mlir/_mlir_libs"/_pto* "${WHEEL_STAGING_DIR}/mlir/_mlir_libs/"

# Copy TileLang resources into the wheel staging tree so wheel installs keep
# the template library and Python DSL available.
echo "Copying TileLang resources..."
rm -rf "${WHEEL_STAGING_DIR}/tilelang_dsl" "${WHEEL_STAGING_DIR}/TileOps"
cp -R "${PTO_INSTALL_DIR}/tilelang_dsl" "${WHEEL_STAGING_DIR}/tilelang_dsl"
cp -R "${PTO_INSTALL_DIR}/share/ptoas/TileOps" "${WHEEL_STAGING_DIR}/TileOps"

# Copy ptodsl into the wheel so it is always shipped with ptoas
rm -rf "${WHEEL_STAGING_DIR}/ptodsl"
cp -R "${PTO_SOURCE_DIR}/ptodsl/ptodsl" "${WHEEL_STAGING_DIR}/ptodsl"

# Copy platform-specific setup.py to package directory.
# On macOS, use setup_mac.py and rename it to setup.py in the build dir.
SETUP_TEMPLATE="${PTO_SOURCE_DIR}/docker/setup.py"
if [ "$(uname -s)" = "Darwin" ] && [ -f "${PTO_SOURCE_DIR}/docker/setup_mac.py" ]; then
  SETUP_TEMPLATE="${PTO_SOURCE_DIR}/docker/setup_mac.py"
fi
echo "Copying $(basename "${SETUP_TEMPLATE}") as setup.py..."
cp "${SETUP_TEMPLATE}" "${WHEEL_STAGING_DIR}/setup.py"

# Determine Python version tag (e.g., cp311, cp312)
PY_VERSION=$("${PYTHON_BIN}" -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
echo "Python version tag: ${PY_VERSION}"

# Build the wheel with version-specific tag
echo "Building wheel..."
cd "${WHEEL_STAGING_DIR}"
if [ -n "${WHEEL_PLAT_NAME:-}" ]; then
  echo "Using wheel platform tag: ${WHEEL_PLAT_NAME}"
  "${PYTHON_BIN}" setup.py bdist_wheel --python-tag "${PY_VERSION}" --plat-name "${WHEEL_PLAT_NAME}" --dist-dir "${WHEEL_DIST_DIR}"
else
  "${PYTHON_BIN}" setup.py bdist_wheel --python-tag "${PY_VERSION}" --dist-dir "${WHEEL_DIST_DIR}"
fi

echo "Wheel created at ${WHEEL_DIST_DIR}/"
ls -la "${WHEEL_DIST_DIR}/"*.whl

EXPECTED_WHEEL_GLOB="${WHEEL_DIST_DIR}/ptoas-${PTOAS_PYTHON_PACKAGE_VERSION}-"*.whl
if ! compgen -G "${EXPECTED_WHEEL_GLOB}" >/dev/null 2>&1; then
  echo "Error: expected wheel matching ${EXPECTED_WHEEL_GLOB}" >&2
  exit 1
fi
