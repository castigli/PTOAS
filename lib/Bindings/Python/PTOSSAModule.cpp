// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOSSAModule.cpp - Python bindings for the pto_ssa dialect ---------===//
//
// Minimal pybind11 extension for the pto_ssa dialect.  This module only
// exposes register_dialect(); all type/attr bindings are inherited from _pto
// (loaded first by pto.py).
//
//===----------------------------------------------------------------------===//

#include "pybind11/pybind11.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir-c/IR.h"

// The generated mlirGetDialectHandle__pto_ssa__() declaration comes from the
// tablegen-generated dialect registration header.
#include "PTO/PTOSSA/IR/PTOSSADialect.h.inc"

namespace py = pybind11;

PYBIND11_MODULE(_pto_ssa, m) {
  m.doc() = "pto_ssa dialect Python bindings (pybind11).";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__pto_ssa__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);
      },
      py::arg("context"), py::arg("load") = true);
}
