# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
PEP 517 build backend for ptoas.

Runs the CMake/Ninja build (assuming LLVM is already built), then delegates
wheel packaging to docker/create_wheel.sh.

Environment variables (all optional):
  LLVM_BUILD_DIR               Path to LLVM build dir
                               (default: /llvm-workspace/llvm-project/build-shared)
  PTO_BUILD_DIR                Path to PTOAS build dir (default: <repo>/build)
  PTO_INSTALL_DIR              Install prefix (default: <repo>/install)
  PTOAS_PYTHON_PACKAGE_VERSION Wheel version override
"""
from __future__ import annotations

import base64
import glob
import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_REPO = Path(__file__).parent.resolve()
_LLVM_BUILD_DIR = Path(
    os.environ.get("LLVM_BUILD_DIR",
                   "/llvm-workspace/llvm-project/build-shared")
)
_PTO_INSTALL_DIR = Path(
    os.environ.get("PTO_INSTALL_DIR", str(_REPO / "install"))
)
_BUILD_DIR = Path(os.environ.get("PTO_BUILD_DIR", str(_REPO / "build")))


def _find_mlir_python_package(prefix: Path) -> Path:
    """Return the MLIR python package directory under *prefix*.

    Handles both build-tree layouts (``tools/mlir/python_packages/mlir_core``)
    and install-prefix layouts (``python_packages/mlir_core``).
    """
    candidates = [
        prefix / "tools" / "mlir" / "python_packages" / "mlir_core",
        prefix / "python_packages" / "mlir_core",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Cannot find MLIR python package under {prefix}. Tried:\n  {tried}"
    )


_MLIR_PY_PKG = _find_mlir_python_package(_LLVM_BUILD_DIR)
_WHEEL_DIST_DIR = _BUILD_DIR / "wheel-dist"


def get_requires_for_build_wheel(config_settings=None):
    return ["setuptools>=68", "wheel", "pybind11<3"]


def get_requires_for_build_editable(config_settings=None):
    return ["setuptools>=68", "wheel", "pybind11<3"]


def get_requires_for_build_sdist(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Return wheel metadata without running the full build."""
    import email.message

    version = os.environ.get("PTOAS_PYTHON_PACKAGE_VERSION", "0.1.0")
    dist_info = Path(metadata_directory) / f"ptoas-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)

    meta = email.message.Message()
    meta["Metadata-Version"] = "2.1"
    meta["Name"] = "ptoas"
    meta["Version"] = version
    meta["Summary"] = "PTO Assembler & Optimizer"
    meta["Requires-Python"] = ">=3.9"
    meta["License"] = "Apache-2.0"
    meta["Requires-Dist"] = "numpy"
    (dist_info / "METADATA").write_text(str(meta))
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: _ptoas_build_backend\n"
        "Root-Is-Purelib: True\nTag: py3-none-any\n"
    )
    return dist_info.name


prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel


def build_sdist(sdist_directory, config_settings=None):
    raise NotImplementedError(
        "ptoas does not support sdist. Use `pip install .` to build a wheel."
    )


def _cmake_configure_and_build(skip_install=False):
    """CMake configure + Ninja build (+ install unless *skip_install*)."""
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)

    pybind11_dir = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--cmakedir"], text=True
    ).strip()

    cmake_cmd = [
        "cmake", "-GNinja",
        f"-S{_REPO}", f"-B{_BUILD_DIR}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DPTO_ENABLE_PYTHON_BINDING=ON",
        f"-DLLVM_DIR={_LLVM_BUILD_DIR}/lib/cmake/llvm",
        f"-DMLIR_DIR={_LLVM_BUILD_DIR}/lib/cmake/mlir",
        f"-DPython3_ROOT_DIR={sys.prefix}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        "-DPython3_FIND_STRATEGY=LOCATION",
        f"-Dpybind11_DIR={pybind11_dir}",
        f"-DMLIR_PYTHON_PACKAGE_DIR={_MLIR_PY_PKG}",
        f"-DCMAKE_INSTALL_PREFIX={_PTO_INSTALL_DIR}",
    ]

    release_version = os.environ.get("PTOAS_RELEASE_VERSION_OVERRIDE", "")
    if release_version:
        cmake_cmd.append(f"-DPTOAS_RELEASE_VERSION_OVERRIDE={release_version}")

    hardening_cache = _REPO / "cmake" / "LinuxHardeningCache.cmake"
    if hardening_cache.exists():
        cmake_cmd.insert(1, f"-C{hardening_cache}")

    subprocess.check_call(cmake_cmd)
    subprocess.check_call(["ninja", "-C", str(_BUILD_DIR)])
    if not skip_install:
        subprocess.check_call(["ninja", "-C", str(_BUILD_DIR), "install"])


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _cmake_configure_and_build()

    env = os.environ.copy()
    env.update({
        "PTO_SOURCE_DIR": str(_REPO),
        "PTO_INSTALL_DIR": str(_PTO_INSTALL_DIR),
        "LLVM_BUILD_DIR": str(_LLVM_BUILD_DIR),
        "PTO_WHEEL_DIST_DIR": str(_WHEEL_DIST_DIR),
    })
    subprocess.check_call(
        ["bash", str(_REPO / "docker" / "create_wheel.sh")],
        env=env,
    )

    wheels = sorted(
        glob.glob(str(_WHEEL_DIST_DIR / "ptoas-*.whl")),
        key=os.path.getmtime,
    )
    if not wheels:
        raise RuntimeError(
            f"No ptoas-*.whl found in {_WHEEL_DIST_DIR} after build."
        )

    wheel_path = Path(wheels[-1])
    dest = Path(wheel_directory) / wheel_path.name
    shutil.copy2(wheel_path, dest)
    return dest.name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """PEP 660 editable install.

    Builds the C++ extensions in-place, then produces a minimal wheel that
    installs a .pth file pointing sys.path at the build tree.  No files are
    copied into site-packages except the .pth file itself.
    """
    _cmake_configure_and_build(skip_install=True)

    version = os.environ.get("PTOAS_PYTHON_PACKAGE_VERSION", "0.1.0")

    # Paths that must be on sys.path for the package to be importable
    pth_paths = [
        # mlir.* namespace + _pto.so (installed there by CMake)
        str(_MLIR_PY_PKG),
        # _pto.so output directory (CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        str(_BUILD_DIR / "python"),
        # ptodsl pure-Python sub-package
        str(_REPO / "ptodsl"),
        # runtime resources installed by CMake (tilelang_dsl, TileOps, etc.)
        str(_PTO_INSTALL_DIR),
    ]

    pth_content = "\n".join(pth_paths) + "\n"
    pth_filename = "ptoas-editable.pth"

    # ---- Build the editable wheel (a zip with .pth + dist-info) ----
    tag = f"py3-none-any"
    wheel_name = f"ptoas-{version}-{tag}.whl"
    wheel_path = Path(wheel_directory) / wheel_name

    dist_info_dir = f"ptoas-{version}.dist-info"

    def _sha256_record(data: bytes) -> str:
        digest = hashlib.sha256(data).digest()
        b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        return f"sha256={b64}"

    pth_bytes = pth_content.encode()
    wheel_meta = (
        "Wheel-Version: 1.0\n"
        "Generator: _ptoas_build_backend\n"
        "Root-Is-Purelib: True\n"
        f"Tag: {tag}\n"
        "Build: editable\n"
    ).encode()
    metadata_content = (
        "Metadata-Version: 2.1\n"
        "Name: ptoas\n"
        f"Version: {version}\n"
        "Summary: PTO Assembler & Optimizer\n"
        "Requires-Python: >=3.9\n"
        "License: Apache-2.0\n"
        "Requires-Dist: numpy\n"
    ).encode()

    record_lines = [
        f"{pth_filename},{_sha256_record(pth_bytes)},{len(pth_bytes)}",
        f"{dist_info_dir}/WHEEL,{_sha256_record(wheel_meta)},{len(wheel_meta)}",
        f"{dist_info_dir}/METADATA,{_sha256_record(metadata_content)},{len(metadata_content)}",
        f"{dist_info_dir}/RECORD,,",
    ]
    record_content = "\n".join(record_lines).encode()

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(pth_filename, pth_bytes)
        zf.writestr(f"{dist_info_dir}/WHEEL", wheel_meta)
        zf.writestr(f"{dist_info_dir}/METADATA", metadata_content)
        zf.writestr(f"{dist_info_dir}/RECORD", record_content)

    return wheel_name
