#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# For quick development, build and install ptoas and its python bindings 
# on top of Docker image https://github.com/learning-chip/agent_docker_npu/pull/8
# assume MLIR is already installed to save time, takes <3min to finish the build of pto extension
#
# Optional env:
#   LLVM_BUILD_DIR   - default: ${LLVM_SOURCE_DIR:-/llvm-workspace/llvm-project}/build-shared
#   PTO_INSTALL_DIR  - default: <repo>/install

set -euo pipefail

PTO_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTO_INSTALL_DIR="${PTO_INSTALL_DIR:-${PTO_SOURCE_DIR}/install}"

LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-/llvm-workspace/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_SOURCE_DIR}/build-shared}"

PY_ROOT="$(python -c 'import sys; print(sys.prefix)')"

for d in "$LLVM_BUILD_DIR/lib/cmake/llvm" "$LLVM_BUILD_DIR/lib/cmake/mlir"; do
  test -d "$d" || { echo "error: missing $d (set LLVM_BUILD_DIR?)" >&2; exit 1; }
done

PYBIND11_DIR="$(python -m pybind11 --cmakedir)"

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
  echo "error: cannot find MLIR python package under ${prefix}" >&2
  echo "Tried:" >&2
  echo "  ${prefix}/tools/mlir/python_packages/mlir_core" >&2
  echo "  ${prefix}/python_packages/mlir_core" >&2
  exit 1
}
MLIR_PY_PKG="$(_find_mlir_python_package "${LLVM_BUILD_DIR}")"

PTOAS_VERSION="${PTOAS_VERSION:-$(python "${PTO_SOURCE_DIR}/.github/scripts/compute_ptoas_version.py" --cmake-file "${PTO_SOURCE_DIR}/CMakeLists.txt" --mode dev)}"

cd "$PTO_SOURCE_DIR"

cmake -C "${PTO_SOURCE_DIR}/cmake/LinuxHardeningCache.cmake" -G Ninja \
  -S . \
  -B build \
  -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
  -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir" \
  -DPython3_ROOT_DIR="${PY_ROOT}" \
  -DPython3_EXECUTABLE=python \
  -DPython3_FIND_STRATEGY=LOCATION \
  -Dpybind11_DIR="${PYBIND11_DIR}" \
  -DMLIR_PYTHON_PACKAGE_DIR="${MLIR_PY_PKG}" \
  -DPTOAS_RELEASE_VERSION_OVERRIDE="${PTOAS_VERSION}" \
  -DCMAKE_INSTALL_PREFIX="${PTO_INSTALL_DIR}"

ninja -C build
ninja -C build install

export PTO_SOURCE_DIR PTO_INSTALL_DIR LLVM_BUILD_DIR
export PTOAS_PYTHON_PACKAGE_VERSION="${PTOAS_PYTHON_PACKAGE_VERSION:-${PTOAS_VERSION}}"
bash "${PTO_SOURCE_DIR}/docker/create_wheel.sh"

shopt -s nullglob
wheels=("${MLIR_PY_PKG}/dist/ptoas-"*.whl)
shopt -u nullglob
((${#wheels[@]} > 0)) || { echo "error: no ptoas-*.whl under ${MLIR_PY_PKG}/dist" >&2; exit 1; }
pip install --force-reinstall "${wheels[0]}"

export PATH="${PTO_SOURCE_DIR}/build/tools/ptoas:${PATH}"
export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

python -c "import mlir.ir"
python -c "from mlir.dialects import pto"

which ptoas

PTOAS_ENV_TMP="${PTO_SOURCE_DIR}/tmp/set_ptoas_env"
mkdir -p "${PTOAS_ENV_TMP}/MatMul" "${PTOAS_ENV_TMP}/Abs"
(cd "${PTO_SOURCE_DIR}/test/samples/MatMul" && python ./tmatmulk.py > "${PTOAS_ENV_TMP}/MatMul/tmatmulk.pto" && ptoas "${PTOAS_ENV_TMP}/MatMul/tmatmulk.pto" -o "${PTOAS_ENV_TMP}/MatMul/tmatmulk.cpp")
(cd "${PTO_SOURCE_DIR}/test/samples/Abs" && python ./abs.py > "${PTOAS_ENV_TMP}/Abs/abs.pto" && ptoas --enable-insert-sync "${PTOAS_ENV_TMP}/Abs/abs.pto" -o "${PTOAS_ENV_TMP}/Abs/abs.cpp")

echo "quick_install.sh: OK"
