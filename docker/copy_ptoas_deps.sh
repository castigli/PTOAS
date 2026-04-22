#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Collect only *.so actually needed by ptoas (transitive closure under /llvm-workspace).
# Expects: LLVM_BUILD_DIR, PTO_INSTALL_DIR, PTOAS_DEPS_DIR, PTO_SOURCE_DIR

set -euo pipefail

export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
PTOAS_BIN="${PTO_SOURCE_DIR}/build/tools/ptoas/ptoas"

remove_rpath() {
  local path="$1"
  if ! readelf -d "$path" 2>/dev/null | grep -Eq '(RPATH|RUNPATH)'; then
    return
  fi
  if command -v patchelf >/dev/null 2>&1; then
    patchelf --remove-rpath "$path"
    return
  fi
  if command -v chrpath >/dev/null 2>&1; then
    chrpath -d "$path"
    return
  fi
  echo "Error: neither patchelf nor chrpath is available to scrub RPATH from ${path}" >&2
  exit 1
}

strip_symbols() {
  local path="$1"
  strip --strip-unneeded "$path"
}

assert_no_rpath() {
  local path="$1"
  if readelf -d "$path" 2>/dev/null | grep -Eq '(RPATH|RUNPATH)'; then
    echo "Error: runtime search path still present in ${path}" >&2
    exit 1
  fi
}

harden_elf() {
  local path="$1"
  remove_rpath "$path"
  strip_symbols "$path"
  assert_no_rpath "$path"
}

copy_so() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local name
  name=$(basename "$f")
  [[ -f "${PTOAS_DEPS_DIR}/${name}" ]] && return 0
  cp -n "$f" "${PTOAS_DEPS_DIR}/" 2>/dev/null || true
  harden_elf "${PTOAS_DEPS_DIR}/${name}"
  while read -r res; do
    copy_so "$res"
  done < <(ldd "$f" 2>/dev/null | awk '/=> \/llvm-workspace\// {print $3}')
}

mkdir -p "$PTOAS_DEPS_DIR"
while read -r res; do
  copy_so "$res"
done < <(ldd "$PTOAS_BIN" 2>/dev/null | awk '/=> \/llvm-workspace\// {print $3}')
