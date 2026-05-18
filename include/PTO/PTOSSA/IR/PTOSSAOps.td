// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOSSAOps.td - SSA-form compute tile ops ---------------*- tablegen -*-===//
//
// Defines the pto_ssa dialect and all its ops.
//
// Every op in this dialect is a functional (SSA) variant of a corresponding
// pto DPS op:
//   - No pre-allocated destination operand (alloc_tile is not needed).
//   - Every op that produces tile data returns a new TileBufType SSA value.
//   - Ops with scratch workspace return the scratch tile as an additional result.
//   - Read-modify-write ops (e.g. tmatmul.acc) take the accumulator as an
//     explicit input (acc_in) and return a new tile as their result.
//   - pto_ssa.tstore is the only op with no tile result (it writes to global
//     memory and has no tile output).
//
// pto_ssa ops carry no memory effects on tile values: they are pure producers.
// Only pto_ssa.tstore has a write effect (to global memory via $dst).
//
// See docs/pto-ssa-dialect.md for the full design rationale and examples.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_SSA_IR_PTOSSAOPS
#define MLIR_DIALECT_PTO_SSA_IR_PTOSSAOPS

// Include the pto dialect ops — this transitively brings in:
//   PTODialect.td, PTOAttrs.td, PTOTypeDefs.td, PTOInterfaces.td,
//   SideEffectInterfaces.td, and all PTODpsType / TileBufType constraints.
include "PTO/IR/PTOOps.td"
include "PTO/PTOSSA/IR/PTOSSADialect.td"

//===----------------------------------------------------------------------===//
// Base class for all pto_ssa ops
//===----------------------------------------------------------------------===//

// All pto_ssa compute ops implement OpPipeInterface so sync-analysis passes
// can query their hardware pipeline, and Pure (no side effects on existing
// SSA values). pto_ssa.tstore overrides Pure with a write effect.
class PTOSSA_Op<string mnemonic, list<Trait> traits = []>
    : Op<PTOSSA_Dialect, mnemonic,
         !listconcat(traits, [OpPipeInterface, Pure])>;

//===----------------------------------------------------------------------===//
// MTE2 — Load / Prefetch
//===----------------------------------------------------------------------===//

// pto_ssa.tload — load from global memory into a new tile.
// Replaces alloc_tile + pto.tload. The result type specifies the target
// on-chip domain (loc=), element type, shape, and layout.
// All optional padding attributes from pto.tload are preserved.
def TLoadSSAOp : PTOSSA_Op<"tload", [AttrSizedOperandSegments]> {
  let summary = "SSA tload: load partition_tensor_view into a new tile.";
  let description = [{
    Functional (SSA) form of pto.tload. Returns a new TileBufType value
    holding the loaded data. No pre-allocated destination needed.

    SSA form:
      %tile = pto_ssa.tload ins(%part : !pto.partition_tensor_view<32x32xf32>)
              : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
  }];

  let arguments = (ins
    PTODpsType:$src,
    OptionalAttr<PTO_PadModeAttr>:$pad_mode,
    Optional<AnyType>:$pad_value,
    Optional<Index>:$left_padding_num,
    Optional<AnyType>:$right_padding_num,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$init_out_buffer,
    Optional<AnyType>:$init_condition
  );

  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict
    (`pad_mode` `=` $pad_mode^)?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`left_padding_num` `=` $left_padding_num^ `:` type($left_padding_num))?
    (`init_out_buffer` `=` $init_out_buffer^)?
    (`right_padding_num` `=` $right_padding_num^ `:` type($right_padding_num))?
    (`init_condition` `=` $init_condition^ `:` type($init_condition))?
    `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
  }];
}

// pto_ssa.tprefetch — prefetch from global memory into a new tile.
def TPrefetchSSAOp : PTOSSA_Op<"tprefetch"> {
  let summary = "SSA tprefetch: prefetch global data into a new tile.";

  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
  }];
}

//===----------------------------------------------------------------------===//
// MTE3 / FIX — Store
//===----------------------------------------------------------------------===//

// pto_ssa.tstore — store a tile back to global memory.
// No tile result: this op writes to global memory ($dst) and is the only
// pto_ssa op with a memory write effect.
def TStoreSSAOp : PTOSSA_Op<"tstore",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "SSA tstore: store tile to partition_tensor_view (no tile result).";
  let description = [{
    Stores data from a tile into global memory. No tile value is produced.
    This is the only pto_ssa op with a memory write side effect.

    SSA form:
      pto_ssa.tstore ins(%tile : !pto.tile_buf<...>)
                     outs(%part : !pto.partition_tensor_view<32x32xf32>)
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<I64>:$preQuantScalar,
    DefaultValuedAttr<PTO_STPhaseAttr, "::mlir::pto::STPhase::Unspecified">:$stPhase,
    DefaultValuedAttr<PTO_AtomicTypeAttr, "::mlir::pto::AtomicType::AtomicNone">:$atomicType,
    DefaultValuedAttr<PTO_ReluPreModeAttr, "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src))
           (`,` $preQuantScalar^ `:` type($preQuantScalar))? `)`
    `outs` `(` $dst `:` qualified(type($dst)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    void getEffects(::mlir::SmallVectorImpl<
        ::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>
        &effects) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(),
                           getDst(),
                           ::mlir::SideEffects::DefaultResource::get());
    }
    ::mlir::pto::PIPE getPipe() {
      // ACC (L0C) -> GM uses FIX pipe; others use MTE3.
      if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(getSrc().getType())) {
        if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                tb.getMemorySpace()))
          if (as.getAddressSpace() == ::mlir::pto::AddressSpace::ACC)
            return ::mlir::pto::PIPE::PIPE_FIX;
      }
      return ::mlir::pto::PIPE::PIPE_MTE3;
    }
  }];
}

//===----------------------------------------------------------------------===//
// MTE1 — Tile Movement
//===----------------------------------------------------------------------===//

// pto_ssa.tmov — move data between on-chip domains, producing a new tile.
// The result type specifies the target domain (loc=) explicitly.
def TMovSSAOp : PTOSSA_Op<"tmov", [AttrSizedOperandSegments]> {
  let summary = "SSA tmov: move tile to a new on-chip domain, returning new tile.";
  let description = [{
    Functional form of pto.tmov. The result type encodes the target domain.

    SSA form:
      %left = pto_ssa.tmov ins(%mat : !pto.tile_buf<loc=mat, ...>)
              : !pto.tile_buf<loc=left, dtype=f32, rows=32, cols=32, ...>
  }];

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$fp,
    Optional<I64>:$preQuantScalar,
    OptionalAttr<PTO_AccToVecModeAttr>:$accToVecMode,
    DefaultValuedAttr<PTO_ReluPreModeAttr, "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode
  );

  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src))
    (`,` $fp^ `:` qualified(type($fp)))?
    (`,` $preQuantScalar^ `:` type($preQuantScalar))?
    `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      auto getAS = [](::mlir::Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
        }
        return std::nullopt;
      };
      auto s = getAS(getSrc().getType());
      auto d = getAS(getResult().getType());
      if (s && d) {
        if (*s == ::mlir::pto::AddressSpace::VEC &&
            *d == ::mlir::pto::AddressSpace::VEC)
          return ::mlir::pto::PIPE::PIPE_V;
        if (*s == ::mlir::pto::AddressSpace::MAT &&
            (*d == ::mlir::pto::AddressSpace::LEFT  ||
             *d == ::mlir::pto::AddressSpace::RIGHT ||
             *d == ::mlir::pto::AddressSpace::BIAS  ||
             *d == ::mlir::pto::AddressSpace::SCALING))
          return ::mlir::pto::PIPE::PIPE_MTE1;
        if (*s == ::mlir::pto::AddressSpace::ACC &&
            (*d == ::mlir::pto::AddressSpace::MAT ||
             *d == ::mlir::pto::AddressSpace::VEC))
          return ::mlir::pto::PIPE::PIPE_FIX;
      }
      return ::mlir::pto::PIPE::PIPE_V;
    }
  }];
}

// pto_ssa.tmov.fp — move/convert using fp (scaling) tile, producing new tile.
def TMovFPSSAOp : PTOSSA_Op<"tmov.fp"> {
  let summary = "SSA tmov.fp: move/convert using fp tile, returning new tile.";

  let arguments = (ins PTODpsType:$src, PTODpsType:$fp);
  let results   = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `:` qualified(type($src)) `,` qualified(type($fp)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE1; }
  }];
}

// pto_ssa.textract — extract sub-tile window, producing new tile.
def TExtractSSAOp : PTOSSA_Op<"textract"> {
  let summary = "SSA textract: extract sub-tile window into new tile.";

  let arguments = (ins PTODpsType:$src, Index:$indexRow, Index:$indexCol);
  let results   = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexRow `,` $indexCol `:`
      qualified(type($src)) `,` type($indexRow) `,` type($indexCol) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      auto getAS = [](::mlir::Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
        }
        return std::nullopt;
      };
      auto s = getAS(getSrc().getType());
      auto d = getAS(getResult().getType());
      if (s && d) {
        if (*s == ::mlir::pto::AddressSpace::MAT &&
            (*d == ::mlir::pto::AddressSpace::LEFT  ||
             *d == ::mlir::pto::AddressSpace::RIGHT ||
             *d == ::mlir::pto::AddressSpace::BIAS  ||
             *d == ::mlir::pto::AddressSpace::SCALING))
          return ::mlir::pto::PIPE::PIPE_MTE1;
        if (*s == ::mlir::pto::AddressSpace::ACC ||
            *s == ::mlir::pto::AddressSpace::VEC)
          return ::mlir::pto::PIPE::PIPE_FIX;
      }
      return ::mlir::pto::PIPE::PIPE_V;
    }
  }];
}

// pto_ssa.textract_fp — extract using fp/scaling tile, producing new tile.
def TExtractFPSSAOp : PTOSSA_Op<"textract_fp"> {
  let summary = "SSA textract_fp: extract acc tile window using fp tile.";

  let arguments = (ins
    PTODpsType:$src, PTODpsType:$fp, Index:$indexRow, Index:$indexCol);
  let results   = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `,` $indexRow `,` $indexCol `:`
      qualified(type($src)) `,` qualified(type($fp)) `,`
      type($indexRow) `,` type($indexCol) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_FIX; }
  }];
}

// pto_ssa.tinsert — insert src sub-tile into $base tile, producing new tile.
// $base is the existing tile being inserted into; the result is a new tile
// with the insertion applied.
def TInsertSSAOp : PTOSSA_Op<"tinsert"> {
  let summary = "SSA tinsert: insert src into base tile at offset, returning new tile.";
  let description = [{
    The $base operand is the existing tile. The result has the same type as
    $base with the sub-tile $src inserted at (indexRow, indexCol).

    SSA form:
      %out = pto_ssa.tinsert ins(%src : ...) base(%base : ...) at[%row, %col]
             : !pto.tile_buf<...>
  }];

  let arguments = (ins
    PTODpsType:$src,
    TileBufType:$base,
    Index:$indexRow,
    Index:$indexCol
  );
  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `base` `(` $base `:` qualified(type($base)) `)`
    `at` `[` $indexRow `,` $indexCol `]`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

// pto_ssa.tinsert_fp — insert using fp/scaling tile, producing new tile.
def TInsertFPSSAOp : PTOSSA_Op<"tinsert_fp"> {
  let summary = "SSA tinsert_fp: insert acc tile window using fp tile.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    TileBufType:$base,
    Index:$indexRow,
    Index:$indexCol
  );
  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `:` qualified(type($src)) `,` qualified(type($fp)) `)`
    `base` `(` $base `:` qualified(type($base)) `)`
    `at` `[` $indexRow `,` $indexCol `]`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_FIX; }
  }];
}

//===----------------------------------------------------------------------===//
// PIPE_M — Matrix ops
//===----------------------------------------------------------------------===//

def TMatmulSSAOp : PTOSSA_Op<"tmatmul"> {
  let summary = "SSA tmatmul: lhs * rhs -> new acc tile.";

  let arguments = (ins
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );
  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TMatmulAccSSAOp : PTOSSA_Op<"tmatmul.acc"> {
  let summary = "SSA tmatmul.acc: acc_in + lhs * rhs -> new acc tile.";
  let description = [{
    $acc_in is the existing accumulator value (explicit SSA input).
    The result is a new TileBufType holding the updated accumulator.
    This makes the accumulation chain explicit in the use-def graph.

    SSA form:
      %acc1 = pto_ssa.tmatmul.acc ins(%acc0, %lhs, %rhs : ...) : !pto.tile_buf<loc=acc, ...>
  }];

  let arguments = (ins
    PTODpsType:$acc_in,
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );
  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $acc_in `,` $lhs `,` $rhs `:`
      qualified(type($acc_in)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TMatmulBiasSSAOp : PTOSSA_Op<"tmatmul.bias"> {
  let summary = "SSA tmatmul.bias: lhs * rhs + bias -> new acc tile.";

  let arguments = (ins PTODpsType:$a, PTODpsType:$b, PTODpsType:$bias);
  let results   = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $a `,` $b `,` $bias `:`
      qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($bias)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TMatmulMxSSAOp : PTOSSA_Op<"tmatmul.mx"> {
  let summary = "SSA tmatmul.mx: mixed-precision matmul -> new acc tile.";

  let arguments = (ins
    PTODpsType:$a, PTODpsType:$a_scale,
    PTODpsType:$b, PTODpsType:$b_scale
  );
  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale `:`
      qualified(type($a)) `,` qualified(type($a_scale)) `,`
      qualified(type($b)) `,` qualified(type($b_scale)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TMatmulMxAccSSAOp : PTOSSA_Op<"tmatmul.mx.acc"> {
  let summary = "SSA tmatmul.mx.acc: c_in + mixed-precision matmul -> new acc tile.";

  let arguments = (ins
    PTODpsType:$c_in,
    PTODpsType:$a, PTODpsType:$a_scale,
    PTODpsType:$b, PTODpsType:$b_scale
  );
  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $c_in `,` $a `,` $a_scale `,` $b `,` $b_scale `:`
      qualified(type($c_in)) `,` qualified(type($a)) `,` qualified(type($a_scale)) `,`
      qualified(type($b)) `,` qualified(type($b_scale)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TMatmulMxBiasSSAOp : PTOSSA_Op<"tmatmul.mx.bias"> {
  let summary = "SSA tmatmul.mx.bias: mixed-precision matmul + bias -> new acc tile.";

  let arguments = (ins
    PTODpsType:$a, PTODpsType:$a_scale,
    PTODpsType:$b, PTODpsType:$b_scale,
    PTODpsType:$bias
  );
  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale `,` $bias `:`
      qualified(type($a)) `,` qualified(type($a_scale)) `,`
      qualified(type($b)) `,` qualified(type($b_scale)) `,`
      qualified(type($bias)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TGemvSSAOp : PTOSSA_Op<"tgemv"> {
  let summary = "SSA tgemv: general matrix-vector multiply -> new acc tile.";

  let arguments = (ins PTODpsType:$lhs, PTODpsType:$rhs);
  let results   = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TGemvAccSSAOp : PTOSSA_Op<"tgemv.acc"> {
  let summary = "SSA tgemv.acc: acc_in + GEMV -> new acc tile.";

  let arguments = (ins PTODpsType:$acc_in, PTODpsType:$lhs, PTODpsType:$rhs);
  let results   = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $acc_in `,` $lhs `,` $rhs `:`
      qualified(type($acc_in)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TGemvBiasSSAOp : PTOSSA_Op<"tgemv.bias"> {
  let summary = "SSA tgemv.bias: GEMV + bias -> new acc tile.";

  let arguments = (ins PTODpsType:$a, PTODpsType:$b, PTODpsType:$bias);
  let results   = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $a `,` $b `,` $bias `:`
      qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($bias)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TGemvMxSSAOp : PTOSSA_Op<"tgemv.mx"> {
  let summary = "SSA tgemv.mx: mixed-precision GEMV -> new acc tile.";

  let arguments = (ins
    PTODpsType:$a, PTODpsType:$a_scale,
    PTODpsType:$b, PTODpsType:$b_scale
  );
  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale `:`
      qualified(type($a)) `,` qualified(type($a_scale)) `,`
      qualified(type($b)) `,` qualified(type($b_scale)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TGemvMxAccSSAOp : PTOSSA_Op<"tgemv.mx.acc"> {
  let summary = "SSA tgemv.mx.acc: c_in + mixed-precision GEMV -> new acc tile.";

  let arguments = (ins
    PTODpsType:$c_in,
    PTODpsType:$a, PTODpsType:$a_scale,
    PTODpsType:$b, PTODpsType:$b_scale
  );
  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $c_in `,` $a `,` $a_scale `,` $b `,` $b_scale `:`
      qualified(type($c_in)) `,` qualified(type($a)) `,` qualified(type($a_scale)) `,`
      qualified(type($b)) `,` qualified(type($b_scale)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TGemvMxBiasSSAOp : PTOSSA_Op<"tgemv.mx.bias"> {
  let summary = "SSA tgemv.mx.bias: mixed-precision GEMV + bias -> new acc tile.";

  let arguments = (ins
    PTODpsType:$a, PTODpsType:$a_scale,
    PTODpsType:$b, PTODpsType:$b_scale,
    PTODpsType:$bias
  );
  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale `,` $bias `:`
      qualified(type($a)) `,` qualified(type($a_scale)) `,`
      qualified(type($b)) `,` qualified(type($b_scale)) `,`
      qualified(type($bias)) `)`
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

//===----------------------------------------------------------------------===//
// PIPE_V — Vector ops (single result, no scratch)
//===----------------------------------------------------------------------===//

def TAbsSSAOp : PTOSSA_Op<"tabs"> {
  let summary = "SSA tabs: elementwise absolute value.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TAddSSAOp : PTOSSA_Op<"tadd"> {
  let summary = "SSA tadd: elementwise add of two tiles.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TAddCSSAOp : PTOSSA_Op<"taddc"> {
  let summary = "SSA taddc: fused multiply-add (src0 + src1 * src2).";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1, PTODpsType:$src2);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src2 `:`
      qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($src2)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TAddSSSAOp : PTOSSA_Op<"tadds"> {
  let summary = "SSA tadds: add tile and scalar.";
  let arguments = (ins PTODpsType:$src, ScalarType:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

// taxpy SSA: $acc_in + $src * $scalar -> new tile.
// $acc_in is the explicit accumulator input (replaces the DPS read-modify-write
// where $dst was both the accumulator and the output).
def TAxpySSAOp : PTOSSA_Op<"taxpy"> {
  let summary = "SSA taxpy: acc_in + src * scalar -> new tile.";
  let description = [{
    $acc_in is the existing accumulator tile (explicit SSA input).
    Result is a new tile holding acc_in + src * scalar.
  }];
  let arguments = (ins PTODpsType:$src, ScalarType:$scalar, PTODpsType:$acc_in);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `,` $acc_in `:`
      qualified(type($src)) `,` type($scalar) `,` qualified(type($acc_in)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TAddSCSSAOp : PTOSSA_Op<"taddsc"> {
  let summary = "SSA taddsc: src0 + scalar * src1 -> new tile.";
  let arguments = (ins PTODpsType:$src0, ScalarType:$scalar, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $scalar `,` $src1 `:`
      qualified(type($src0)) `,` type($scalar) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TAndSSAOp : PTOSSA_Op<"tand"> {
  let summary = "SSA tand: elementwise bitwise AND.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TAndSSSAOp : PTOSSA_Op<"tands"> {
  let summary = "SSA tands: elementwise bitwise AND with scalar.";
  let arguments = (ins PTODpsType:$src, AnySignlessInteger:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TCISSAOp : PTOSSA_Op<"tci"> {
  let summary = "SSA tci: generate column-index tile.";
  let arguments = (ins
    AnyInteger:$S,
    DefaultValuedAttr<BoolAttr, "false">:$descending
  );
  let results = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $S `:` type($S) `)` attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TTriSSAOp : PTOSSA_Op<"ttri"> {
  let summary = "SSA ttri: generate triangular mask tile.";
  let arguments = (ins
    AnyInteger:$diagonal,
    DefaultValuedAttr<BoolAttr, "false">:$upperOrLower
  );
  let results = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $diagonal `:` type($diagonal) `)` attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TCmpSSAOp : PTOSSA_Op<"tcmp"> {
  let summary = "SSA tcmp: elementwise compare of two tiles -> mask tile.";
  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    OptionalAttr<PTO_CmpModeAttr>:$cmpMode
  );
  let results = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TCmpSSSAOp : PTOSSA_Op<"tcmps"> {
  let summary = "SSA tcmps: compare tile with scalar -> mask tile.";
  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTO_CmpModeAttr:$cmpMode
  );
  let results = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColExpandSSAOp : PTOSSA_Op<"tcolexpand"> {
  let summary = "SSA tcolexpand: broadcast column vector across all columns.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

// Helper macro-like multiline def for tcolexpand binary ops
def TColExpandAddSSAOp : PTOSSA_Op<"tcolexpandadd"> {
  let summary = "SSA tcolexpandadd: broadcast-add column vector to tile.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColExpandMulSSAOp : PTOSSA_Op<"tcolexpandmul"> {
  let summary = "SSA tcolexpandmul: broadcast-multiply column vector with tile.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColExpandDivSSAOp : PTOSSA_Op<"tcolexpanddiv"> {
  let summary = "SSA tcolexpanddiv: broadcast-divide tile by column vector.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColExpandSubSSAOp : PTOSSA_Op<"tcolexpandsub"> {
  let summary = "SSA tcolexpandsub: broadcast-subtract column vector from tile.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColExpandExpdifSSAOp : PTOSSA_Op<"tcolexpandexpdif"> {
  let summary = "SSA tcolexpandexpdif: broadcast exp-difference with column vector.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColExpandMaxSSAOp : PTOSSA_Op<"tcolexpandmax"> {
  let summary = "SSA tcolexpandmax: elementwise max with broadcast column vector.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColExpandMinSSAOp : PTOSSA_Op<"tcolexpandmin"> {
  let summary = "SSA tcolexpandmin: elementwise min with broadcast column vector.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColMaxSSAOp : PTOSSA_Op<"tcolmax"> {
  let summary = "SSA tcolmax: reduce each column to its max element.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColMinSSAOp : PTOSSA_Op<"tcolmin"> {
  let summary = "SSA tcolmin: reduce each column to its min element.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColProdSSAOp : PTOSSA_Op<"tcolprod"> {
  let summary = "SSA tcolprod: reduce each column by multiplication.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TConcatSSAOp : PTOSSA_Op<"tconcat"> {
  let summary = "SSA tconcat: concatenate two tiles along columns.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TCvtSSAOp : PTOSSA_Op<"tcvt"> {
  let summary = "SSA tcvt: elementwise type conversion.";
  let arguments = (ins
    PTODpsType:$src,
    DefaultValuedAttr<PTO_RoundModeAttr, "::mlir::pto::RoundMode::CAST_RINT">:$rmode,
    DefaultValuedAttr<PTO_SaturationModeAttr, "::mlir::pto::SaturationMode::OFF">:$sat_mode
  );
  let results = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)` attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TDequantSSAOp : PTOSSA_Op<"tdequant"> {
  let summary = "SSA tdequant: dequantize integer tile using scale and offset.";
  let arguments = (ins PTODpsType:$src, PTODpsType:$scale, PTODpsType:$offset);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scale `,` $offset `:`
      qualified(type($src)) `,` qualified(type($scale)) `,` qualified(type($offset)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TDivSSAOp : PTOSSA_Op<"tdiv"> {
  let summary = "SSA tdiv: elementwise division of two tiles.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TDivSSSAOp : PTOSSA_Op<"tdivs"> {
  let summary = "SSA tdivs: divide tile by scalar.";
  let arguments = (ins AnyType:$src, AnyType:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` type($src) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TExpSSAOp : PTOSSA_Op<"texp"> {
  let summary = "SSA texp: elementwise exp.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TExpandsSSAOp : PTOSSA_Op<"texpands"> {
  let summary = "SSA texpands: fill tile with exp(scalar).";
  let arguments = (ins ScalarType:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $scalar `:` type($scalar) `)` attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TFillPadSSAOp : PTOSSA_Op<"tfillpad"> {
  let summary = "SSA tfillpad: copy src and fill pad region from PadVal.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TFillPadExpandSSAOp : PTOSSA_Op<"tfillpad_expand"> {
  let summary = "SSA tfillpad_expand: copy src and fill+expand pad region.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TFillPadInplaceSSAOp : PTOSSA_Op<"tfillpad_inplace"> {
  let summary = "SSA tfillpad_inplace: fill pad region of src tile.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TGatherBSSAOp : PTOSSA_Op<"tgatherb"> {
  let summary = "SSA tgatherb: gather using byte offsets.";
  let arguments = (ins PTODpsType:$src, PTODpsType:$offsets);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $offsets `:` qualified(type($src)) `,` type($offsets) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TLogSSAOp : PTOSSA_Op<"tlog"> {
  let summary = "SSA tlog: elementwise natural logarithm.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TLReluSSAOp : PTOSSA_Op<"tlrelu"> {
  let summary = "SSA tlrelu: leaky ReLU with fixed slope.";
  let arguments = (ins PTODpsType:$src, F32:$slope);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $slope `:` qualified(type($src)) `,` type($slope) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TMaxSSAOp : PTOSSA_Op<"tmax"> {
  let summary = "SSA tmax: elementwise max of two tiles.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TMaxSSSAOp : PTOSSA_Op<"tmaxs"> {
  let summary = "SSA tmaxs: elementwise max of tile and scalar.";
  let arguments = (ins PTODpsType:$src, ScalarType:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TMinSSAOp : PTOSSA_Op<"tmin"> {
  let summary = "SSA tmin: elementwise min of two tiles.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TMinSSSAOp : PTOSSA_Op<"tmins"> {
  let summary = "SSA tmins: elementwise min of tile and scalar.";
  let arguments = (ins PTODpsType:$src, ScalarType:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TMulSSAOp : PTOSSA_Op<"tmul"> {
  let summary = "SSA tmul: elementwise multiplication of two tiles.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TMulSSSAOp : PTOSSA_Op<"tmuls"> {
  let summary = "SSA tmuls: multiply tile by scalar.";
  let arguments = (ins PTODpsType:$src0, ScalarType:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $scalar `:` qualified(type($src0)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TNegSSAOp : PTOSSA_Op<"tneg"> {
  let summary = "SSA tneg: elementwise negation.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TNotSSAOp : PTOSSA_Op<"tnot"> {
  let summary = "SSA tnot: elementwise bitwise NOT.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TOrSSAOp : PTOSSA_Op<"tor"> {
  let summary = "SSA tor: elementwise bitwise OR.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TOrSSSAOp : PTOSSA_Op<"tors"> {
  let summary = "SSA tors: elementwise bitwise OR with scalar.";
  let arguments = (ins PTODpsType:$src, AnySignlessInteger:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TPartAddSSAOp : PTOSSA_Op<"tpartadd"> {
  let summary = "SSA tpartadd: partial elementwise add.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TPartMaxSSAOp : PTOSSA_Op<"tpartmax"> {
  let summary = "SSA tpartmax: partial elementwise max.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TPartMinSSAOp : PTOSSA_Op<"tpartmin"> {
  let summary = "SSA tpartmin: partial elementwise min.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TPartMulSSAOp : PTOSSA_Op<"tpartmul"> {
  let summary = "SSA tpartmul: partial elementwise multiply.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TQuantSSAOp : PTOSSA_Op<"tquant"> {
  let summary = "SSA tquant: quantize tile using fp scaling tile.";
  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    Optional<PTODpsType>:$offset,
    PTO_QuantTypeAttr:$quant_type
  );
  let results = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $fp (`,` $offset^)? `:`
      qualified(type($src)) `,` qualified(type($fp))
      (`,` qualified(type($offset))^)? `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRecipSSAOp : PTOSSA_Op<"trecip"> {
  let summary = "SSA trecip: elementwise reciprocal.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TReluSSAOp : PTOSSA_Op<"trelu"> {
  let summary = "SSA trelu: elementwise ReLU.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowExpandSSAOp : PTOSSA_Op<"trowexpand"> {
  let summary = "SSA trowexpand: broadcast row vector across all rows.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowExpandAddSSAOp : PTOSSA_Op<"trowexpandadd"> {
  let summary = "SSA trowexpandadd: broadcast-add row vector to tile.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TShlSSAOp : PTOSSA_Op<"tshl"> {
  let summary = "SSA tshl: elementwise left shift.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TShlSSSAOp : PTOSSA_Op<"tshls"> {
  let summary = "SSA tshls: left shift tile by scalar.";
  let arguments = (ins PTODpsType:$src, AnySignlessInteger:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TShrSSAOp : PTOSSA_Op<"tshr"> {
  let summary = "SSA tshr: elementwise right shift.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TShrSSSAOp : PTOSSA_Op<"tshrs"> {
  let summary = "SSA tshrs: right shift tile by scalar.";
  let arguments = (ins PTODpsType:$src, AnySignlessInteger:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TSqrtSSAOp : PTOSSA_Op<"tsqrt"> {
  let summary = "SSA tsqrt: elementwise square root.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TSubSSAOp : PTOSSA_Op<"tsub"> {
  let summary = "SSA tsub: elementwise subtraction.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TSubCSSAOp : PTOSSA_Op<"tsubc"> {
  let summary = "SSA tsubc: fused sub-multiply (src0 - src1 * src2).";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1, PTODpsType:$src2);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src2 `:`
      qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($src2)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TSubSSSAOp : PTOSSA_Op<"tsubs"> {
  let summary = "SSA tsubs: subtract scalar from tile.";
  let arguments = (ins PTODpsType:$src, ScalarType:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TSubSCSSAOp : PTOSSA_Op<"tsubsc"> {
  let summary = "SSA tsubsc: src0 - scalar * src1 -> new tile.";
  let arguments = (ins PTODpsType:$src0, ScalarType:$scalar, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $scalar `,` $src1 `:`
      qualified(type($src0)) `,` type($scalar) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TFModSSAOp : PTOSSA_Op<"tfmod"> {
  let summary = "SSA tfmod: elementwise floating-point modulo.";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TFModSSSAOp : PTOSSA_Op<"tfmods"> {
  let summary = "SSA tfmods: elementwise fmod with scalar.";
  let arguments = (ins PTODpsType:$src, ScalarType:$scalar);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRandomSSAOp : PTOSSA_Op<"trandom"> {
  let summary = "SSA trandom: generate random tile from key/counter words.";
  let arguments = (ins
    AnySignlessInteger:$key0,
    AnySignlessInteger:$key1,
    AnySignlessInteger:$counter0,
    AnySignlessInteger:$counter1,
    AnySignlessInteger:$counter2,
    AnySignlessInteger:$counter3,
    DefaultValuedAttr<I32Attr, "10">:$rounds
  );
  let results = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $key0 `,` $key1 `,` $counter0 `,` $counter1 `,` $counter2 `,` $counter3
    attr-dict
    `:` type($key0) `,` type($key1) `,` type($counter0) `,` type($counter1)
        `,` type($counter2) `,` type($counter3) `)`
    `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def THistogramSSAOp : PTOSSA_Op<"thistogram"> {
  let summary = "SSA thistogram: compute histogram of tile values.";
  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$idx,
    DefaultValuedAttr<BoolAttr, "false">:$isMSB
  );
  let results = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $idx `:` qualified(type($src)) `,` qualified(type($idx)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def MGatherSSAOp : PTOSSA_Op<"mgather"> {
  let summary = "SSA mgather: gather from global memory using index tile.";
  let arguments = (ins
    PTODpsType:$mem,
    PTODpsType:$idx,
    DefaultValuedAttr<BoolAttr, "false">:$gatherOob
  );
  let results = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $mem `,` $idx `:` qualified(type($mem)) `,` qualified(type($idx)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def MScatterSSAOp : PTOSSA_Op<"mscatter"> {
  let summary = "SSA mscatter: scatter tile to global memory using index tile.";
  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$idx,
    PTODpsType:$mem,
    OptionalAttr<PTO_AtomicTypeAttr>:$scatterAtomicOp,
    DefaultValuedAttr<BoolAttr, "false">:$scatterOob
  );
  let results = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $idx `,` $mem `:`
      qualified(type($src)) `,` qualified(type($idx)) `,` qualified(type($mem)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

//===----------------------------------------------------------------------===//
// PIPE_V — Vector ops with $tmp_out scratch result
//===----------------------------------------------------------------------===//

def TTTransSSAOp : PTOSSA_Op<"ttrans"> {
  let summary = "SSA ttrans: transpose tile, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColArgMaxSSAOp : PTOSSA_Op<"tcolargmax"> {
  let summary = "SSA tcolargmax: column-wise argmax, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColArgMinSSAOp : PTOSSA_Op<"tcolargmin"> {
  let summary = "SSA tcolargmin: column-wise argmin, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TColSumSSAOp : PTOSSA_Op<"tcolsum"> {
  let summary = "SSA tcolsum: column-wise sum, returning (result, tmp_out).";
  let arguments = (ins
    PTODpsType:$src,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$isBinary
  );
  let results = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)` attr-dict
    `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowMaxSSAOp : PTOSSA_Op<"trowmax"> {
  let summary = "SSA trowmax: row-wise max, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowArgMaxSSAOp : PTOSSA_Op<"trowargmax"> {
  let summary = "SSA trowargmax: row-wise argmax, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowMinSSAOp : PTOSSA_Op<"trowmin"> {
  let summary = "SSA trowmin: row-wise min, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowArgMinSSAOp : PTOSSA_Op<"trowargmin"> {
  let summary = "SSA trowargmin: row-wise argmin, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowSumSSAOp : PTOSSA_Op<"trowsum"> {
  let summary = "SSA trowsum: row-wise sum, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowProdSSAOp : PTOSSA_Op<"trowprod"> {
  let summary = "SSA trowprod: row-wise product, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRsqrtSSAOp : PTOSSA_Op<"trsqrt"> {
  let summary = "SSA trsqrt: reciprocal sqrt, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TGatherSSAOp : PTOSSA_Op<"tgather", [AttrSizedOperandSegments]> {
  let summary = "SSA tgather: gather/select using index+tmp, scalar-compare, or mask.";
  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$cdst,
    Optional<PTODpsType>:$indices,
    Optional<ScalarType>:$kValue,
    OptionalAttr<PTO_MaskPatternAttr>:$maskPattern,
    OptionalAttr<PTO_CmpModeAttr>:$cmpMode,
    OptionalAttr<I32Attr>:$offset
  );
  let results = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src))
      (`,` `cdst` `=` $cdst^ `:` qualified(type($cdst)))?
      (`,` `indices` `=` $indices^ `:` qualified(type($indices)))?
      (`,` `kValue` `=` $kValue^ `:` type($kValue))?
    `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TSelSSAOp : PTOSSA_Op<"tsel"> {
  let summary = "SSA tsel: select between two tiles using mask, returning (result, tmp_out).";
  let arguments = (ins
    PTODpsType:$mask,
    PTODpsType:$src0,
    PTODpsType:$src1
  );
  let results = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $mask `,` $src0 `,` $src1 `:`
      qualified(type($mask)) `,` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TSelSSSAOp : PTOSSA_Op<"tsels"> {
  let summary = "SSA tsels: global-select using scalar mode, returning (result, tmp_out).";
  let arguments = (ins
    PTODpsType:$mask,
    PTODpsType:$src,
    ScalarType:$scalar
  );
  let results = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $mask `,` $src `,` $scalar `:`
      qualified(type($mask)) `,` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TPReluSSAOp : PTOSSA_Op<"tprelu"> {
  let summary = "SSA tprelu: parametric ReLU, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRemSSAOp : PTOSSA_Op<"trem"> {
  let summary = "SSA trem: elementwise remainder, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRemSSSAOp : PTOSSA_Op<"trems"> {
  let summary = "SSA trems: remainder with scalar, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src, ScalarType:$scalar);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TSort32SSAOp : PTOSSA_Op<"tsort32"> {
  let summary = "SSA tsort32: sort 32-element blocks, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src, PTODpsType:$idx);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `,` $idx `:` qualified(type($src)) `,` qualified(type($idx)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TXorSSAOp : PTOSSA_Op<"txor"> {
  let summary = "SSA txor: elementwise bitwise XOR, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TXorSSSAOp : PTOSSA_Op<"txors"> {
  let summary = "SSA txors: bitwise XOR with scalar, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src, AnySignlessInteger:$scalar);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowExpandDivSSAOp : PTOSSA_Op<"trowexpanddiv"> {
  let summary = "SSA trowexpanddiv: broadcast-divide tile by row vector, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowExpandMulSSAOp : PTOSSA_Op<"trowexpandmul"> {
  let summary = "SSA trowexpandmul: broadcast-multiply tile by row vector, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowExpandSubSSAOp : PTOSSA_Op<"trowexpandsub"> {
  let summary = "SSA trowexpandsub: broadcast-subtract row vector from tile, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowExpandExpdifSSAOp : PTOSSA_Op<"trowexpandexpdif"> {
  let summary = "SSA trowexpandexpdif: broadcast exp-diff with row vector, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowExpandMaxSSAOp : PTOSSA_Op<"trowexpandmax"> {
  let summary = "SSA trowexpandmax: elementwise max with broadcast row vector, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TRowExpandMinSSAOp : PTOSSA_Op<"trowexpandmin"> {
  let summary = "SSA trowexpandmin: elementwise min with broadcast row vector, returning (result, tmp_out).";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1);
  let results   = (outs TileBufType:$result, TileBufType:$tmp_out);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($tmp_out))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

//===----------------------------------------------------------------------===//
// PIPE_V — Dual-output ops (two independent tile results)
//===----------------------------------------------------------------------===//

def TPartArgMaxSSAOp : PTOSSA_Op<"tpartargmax"> {
  let summary = "SSA tpartargmax: partial max with values and indices, returning (result, result_idx).";
  let arguments = (ins
    PTODpsType:$src0, PTODpsType:$src1,
    PTODpsType:$src0Idx, PTODpsType:$src1Idx
  );
  let results = (outs TileBufType:$result, TileBufType:$result_idx);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src0Idx `,` $src1Idx `:`
      qualified(type($src0)) `,` qualified(type($src1)) `,`
      qualified(type($src0Idx)) `,` qualified(type($src1Idx)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($result_idx))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TPartArgMinSSAOp : PTOSSA_Op<"tpartargmin"> {
  let summary = "SSA tpartargmin: partial min with values and indices, returning (result, result_idx).";
  let arguments = (ins
    PTODpsType:$src0, PTODpsType:$src1,
    PTODpsType:$src0Idx, PTODpsType:$src1Idx
  );
  let results = (outs TileBufType:$result, TileBufType:$result_idx);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src0Idx `,` $src1Idx `:`
      qualified(type($src0)) `,` qualified(type($src1)) `,`
      qualified(type($src0Idx)) `,` qualified(type($src1Idx)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($result_idx))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TConcatidxSSAOp : PTOSSA_Op<"tconcatidx"> {
  let summary = "SSA tconcatidx: indexed column concat, returning (result, result_idx).";
  let arguments = (ins
    PTODpsType:$src0, PTODpsType:$src1,
    PTODpsType:$src0Idx, PTODpsType:$src1Idx
  );
  let results = (outs TileBufType:$result, TileBufType:$result_idx);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src0Idx `,` $src1Idx `:`
      qualified(type($src0)) `,` qualified(type($src1)) `,`
      qualified(type($src0Idx)) `,` qualified(type($src1Idx)) `)`
    attr-dict `:` qualified(type($result)) `,` qualified(type($result_idx))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

//===----------------------------------------------------------------------===//
// PIPE_V — Variadic ops
//===----------------------------------------------------------------------===//

// tmrgsort SSA: variadic merge sort.
// $srcs are the input tiles (2-4 for format2, 1 for format1).
// All output tiles are returned as variadic results; the caller discards
// any tmp slots they don't need.
def TMrgSortSSAOp : PTOSSA_Op<"tmrgsort"> {
  let summary = "SSA tmrgsort: multi-way merge sort on tile groups.";
  let description = [{
    SSA form of pto.tmrgsort. Returns one result tile per input src tile.
    Format 1 (single src + blockLen): blockLen attribute required.
    Format 2 (2-4 srcs): returns results + one excuted result tile.
  }];
  let arguments = (ins
    Variadic<PTODpsType>:$srcs,
    Optional<AnyInteger>:$blockLen,
    DefaultValuedAttr<BoolAttr, "false">:$exhausted
  );
  let results = (outs Variadic<TileBufType>:$results);
  let assemblyFormat = [{
    `ins` `(` $srcs `:` type($srcs) `)` attr-dict `:` type($results)
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

//===----------------------------------------------------------------------===//
// PIPE_S — Scalar / misc
//===----------------------------------------------------------------------===//

// pto_ssa.tsetval — set value at offset in a tile, producing new tile.
// $base is the existing tile; $offset and $val specify what to write.
// Result is a new tile with the value set.
def TSetValSSAOp : PTOSSA_Op<"tsetval"> {
  let summary = "SSA tsetval: set scalar at offset in tile, returning new tile.";
  let description = [{
    $base is the existing tile being updated.
    Result is a new tile with $val written at $offset.
  }];
  let arguments = (ins
    TileBufType:$base,
    Index:$offset,
    ScalarType:$val
  );
  let results = (outs TileBufType:$result);
  let assemblyFormat = [{
    `base` `(` $base `:` qualified(type($base)) `)`
    `at` `[` $offset `]` `=` $val `:` type($val)
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

def TScatterSSAOp : PTOSSA_Op<"tscatter"> {
  let summary = "SSA tscatter: scatter elements using index tile.";
  let arguments = (ins PTODpsType:$src, PTODpsType:$indexes);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `,` $indexes `:` qualified(type($src)) `,` qualified(type($indexes)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

def TGetScaleAddrSSAOp : PTOSSA_Op<"tget_scale_addr"> {
  let summary = "SSA tget_scale_addr: get scale address tile.";
  let arguments = (ins PTODpsType:$src);
  let results   = (outs TileBufType:$result);
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

#endif // MLIR_DIALECT_PTO_SSA_IR_PTOSSAOPS
