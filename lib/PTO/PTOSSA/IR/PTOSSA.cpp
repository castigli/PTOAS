// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOSSA.cpp - PTO SSA Dialect implementation ------------------------===//
//
// Implements PTOSSADialect::initialize() and op definitions for the pto_ssa
// dialect. All compute ops are Pure (no side effects). TStoreSSAOp overrides
// this with a Write effect on $dst, declared inline via extraClassDeclaration
// in PTOSSAOps.td.
//
//===----------------------------------------------------------------------===//

#include "PTO/PTOSSA/IR/PTOSSA.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

// Pull in generated dialect boilerplate
#include "PTO/PTOSSA/IR/PTOSSADialect.cpp.inc"

// Pull in generated op definitions
#define GET_OP_CLASSES
#include "PTO/PTOSSA/IR/PTOSSAOps.cpp.inc"

using namespace mlir;
using namespace mlir::pto_ssa;

//===----------------------------------------------------------------------===//
// PTOSSADialect
//===----------------------------------------------------------------------===//

void PTOSSADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PTO/PTOSSA/IR/PTOSSAOps.cpp.inc"
  >();
}
