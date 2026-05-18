# PTO SSA Dialect

The `pto_ssa` dialect provides functional (SSA-form) variants of all PTO compute tile ops.  
Every op returns one or more new `TileBufType` tile values instead of writing into a  
pre-allocated destination buffer.

## Motivation

The DPS (destination-passing-style) `pto` ops require the caller to `alloc_tile` a  
destination buffer before each compute op.  This couples memory allocation with  
computation and complicates scheduling and CSE.  `pto_ssa` removes that coupling:  
the producing op is itself the definition point of the tile value.

## Design principles

- **No pre-allocated destination**: `alloc_tile` is not used in `pto_ssa` code.
- **Pure ops**: all compute ops have no memory side effects on SSA tile values.
- **Scratch workspace**: DPS ops with an optional `$tmp` scratch buffer expose it as  
  an additional `TileBufType` result in the SSA variant.
- **Read-modify-write**: `tmatmul.acc` takes an explicit `$acc_in` SSA tile and  
  returns a new tile, replacing the DPS read-modify-write on `$dst`.
- **Single store op with a write effect**: `pto_ssa.tstore` writes to global memory  
  (`$dst : !pto.partition_tensor_view<...>`) and is the only op that carries a  
  `MemoryEffects::Write` effect.

## Coexistence with `pto`

`pto_ssa` is a separate dialect (`pto_ssa.*` mnemonic prefix, C++ namespace  
`::mlir::pto_ssa`).  It reuses all types, attrs, and enums from the `pto` dialect  
(`TileBufType`, `PartitionTensorViewType`, `AddressSpace`, etc.) and requires `pto`  
to be registered in the same `MLIRContext`.

## Op table

| Op | Inputs | Results | Pipe | Notes |
|----|--------|---------|------|-------|
| `tload` | `src: PTODpsType` + optional pad attrs | `result: TileBufType` | MTE2 | Replaces `alloc_tile + pto.tload` |
| `tprefetch` | `src: PTODpsType` | `result: TileBufType` | MTE2 | |
| `tstore` | `src: PTODpsType`, `dst: PTODpsType` | _(none)_ | MTE3/FIX | Only op with Write effect |
| `tmov` | `src: TileBufType` | `result: TileBufType` | MTE1/MTE3 | Determined by src/dst AddressSpace |
| `textract` | `src: TileBufType`, index attrs | `result: TileBufType` | MTE1/MTE3 | |
| `tinsert` | `base: TileBufType`, `val: TileBufType`, offset | `result: TileBufType` | MTE1/MTE3 | |
| `tmatmul` | `left: TileBufType`, `right: TileBufType` | `result: TileBufType` | PIPE_M | |
| `tmatmul_acc` | `left`, `right`, `acc_in: TileBufType` | `result: TileBufType` | PIPE_M | Explicit accumulator input |
| `tvadd` | `lhs: TileBufType`, `rhs: TileBufType` | `result: TileBufType` | PIPE_V | |
| `tvsub` | `lhs`, `rhs` | `result` | PIPE_V | |
| `tvmul` | `lhs`, `rhs` | `result` | PIPE_V | |
| `tvmax` | `lhs`, `rhs` | `result` | PIPE_V | |
| `tvmin` | `lhs`, `rhs` | `result` | PIPE_V | |
| `taxpy` | `x: TileBufType`, `alpha`, `acc_in: TileBufType` | `result` | PIPE_V | `acc_in` replaces DPS dst |
| `tcolsum` | `src: TileBufType` | `result`, `tmp_out: TileBufType` | PIPE_V | Scratch always returned |
| `tquant` | `src: TileBufType`, optional `offset` | `result` | PIPE_V | |
| `tgather` | `src`, indices, k/maskPattern attrs | `result`, `tmp_out` | PIPE_V | |
| `tmrgsort` | variadic `srcs: TileBufType` | variadic `results` | PIPE_S | |
| `trowexpand` | `src`, pad attrs | `result`, `tmp_out` | PIPE_V | |

> This table is illustrative; see `include/PTO/PTOSSA/IR/PTOSSAOps.td` for the  
> authoritative op list (~105 ops covering MTE2, MTE3, MTE1, PIPE_M, PIPE_V, PIPE_S).

## Python usage

```python
from mlir.ir import Context, Module
from mlir.dialects import pto

with Context() as ctx:
    pto.register_dialect(ctx)       # pto dialect
    pto.register_ssa_dialect(ctx)   # pto_ssa dialect

    mod = Module.parse("""
    module {
      func.func @example(%src: !pto.partition_tensor_view<32x32xf16>,
                         %dst: !pto.partition_tensor_view<32x32xf16>) {
        %t = pto_ssa.tload ins(%src : !pto.partition_tensor_view<32x32xf16>)
             : !pto.tile_buf<[32, 32], f16, VEC>
        pto_ssa.tstore ins(%t : !pto.tile_buf<[32, 32], f16, VEC>)
                      outs(%dst : !pto.partition_tensor_view<32x32xf16>)
        return
      }
    }
    """, ctx)
```

## Lit testing

Use `ptoas --parse-only` to round-trip `pto_ssa` IR without running the full  
lowering pipeline (which targets C++ emission and does not handle `pto_ssa` ops):

```
// RUN: ptoas --parse-only %s | FileCheck %s
```
