// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_SSA_IR_PTOSSADIALECT
#define MLIR_DIALECT_PTO_SSA_IR_PTOSSADIALECT

include "mlir/IR/OpBase.td"

def PTOSSA_Dialect : Dialect {
  let name = "pto_ssa";
  let cppNamespace = "::mlir::pto_ssa";
  let summary = "SSA-form compute tile ops for the PTO dialect.";
  let description = [{
    pto_ssa provides functional (SSA) variants of all PTO compute tile ops.
    Every op returns one or more new tile values instead of writing into a
    pre-allocated destination buffer. See docs/pto-ssa-dialect.md.

    pto_ssa ops use types defined in the pto dialect (TileBufType,
    PartitionTensorViewType, etc.) and depend on the pto dialect being
    registered. The two dialects are designed to coexist: DPS pto ops and
    SSA pto_ssa ops can appear in the same function.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = [
    "::mlir::pto::PTODialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect"
  ];
}

#endif // MLIR_DIALECT_PTO_SSA_IR_PTOSSADIALECT
