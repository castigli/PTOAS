// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOSSA.h - PTO SSA Dialect ----------------------------*- C++ -*-===//
//
// This file defines the PTO SSA dialect — SSA-form functional variants of
// all PTO compute tile ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_SSA_IR_PTOSSA_H_
#define MLIR_DIALECT_PTO_SSA_IR_PTOSSA_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Pull in all PTO types, attrs, enums, and interfaces transitively
#include "PTO/IR/PTO.h"

//===----------------------------------------------------------------------===//
// PTO SSA Dialect
//===----------------------------------------------------------------------===//

#include "PTO/PTOSSA/IR/PTOSSADialect.h"

//===----------------------------------------------------------------------===//
// PTO SSA Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "PTO/PTOSSA/IR/PTOSSAOps.h.inc"

#endif // MLIR_DIALECT_PTO_SSA_IR_PTOSSA_H_
