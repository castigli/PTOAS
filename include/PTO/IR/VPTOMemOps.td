// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOMemOps.td - PTO raw ubuf memory ops (a2a3) -------*- tablegen -*-===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOMEMOPS
#define MLIR_DIALECT_PTO_IR_VPTOMEMOPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "PTO/IR/VPTOInterfaces.td"

class PTO_UBufBinaryOp<string mnemonic, list<Trait> traits = []>
    : PTO_Op<mnemonic,
      !listconcat(traits, [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>])> {
  let summary = "Raw ubuf binary operation mapping to llvm.hivm intrinsics.";
  let description = [{
    Lowers to `llvm.hivm` intrinsics for the a2a3 (dav-m200-vec) backend.
    Operates directly on ubuf pointers rather than vreg values.

    The repeat / block-stride / repeat-stride operands are packed into a single
    `i64` config during lowering, following the bit layout defined in
    `cce_aicore_intrinsics_3101.h`.
  }];

  let arguments = (ins
    PTO_BufferType:$dst,
    PTO_BufferType:$src0,
    PTO_BufferType:$src1,
    I64:$repeat,
    I64:$dstBlockStride,
    I64:$src0BlockStride,
    I64:$src1BlockStride,
    I64:$dstRepeatStride,
    I64:$src0RepeatStride,
    I64:$src1RepeatStride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $dst `,` $src0 `,` $src1 `,`
    $repeat `,` $dstBlockStride `,` $src0BlockStride `,`
    $src1BlockStride `,` $dstRepeatStride `,` $src0RepeatStride `,`
    $src1RepeatStride
    attr-dict `:` type($dst) `,` type($src0) `,` type($src1) `,`
    type($repeat) `,` type($dstBlockStride) `,` type($src0BlockStride) `,`
    type($src1BlockStride) `,` type($dstRepeatStride) `,`
    type($src0RepeatStride) `,` type($src1RepeatStride)
  }];
}

def PTO_UBVaddOp : PTO_UBufBinaryOp<"ub.vadd">;

#endif // MLIR_DIALECT_PTO_IR_VPTOMEMOPS
