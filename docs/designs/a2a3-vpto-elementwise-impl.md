# A2/A3 VPTO Elementwise Lowering — Implementation Notes

## Op Coverage

### Binary Tile-Tile Ops (direct intrinsics)

| PTO Op | UB Op | Intrinsic | dtypes | Status |
|---|---|---|---|---|
| `tadd` | `ub.vadd` | `VADD.f32/f16/s16/s32` | f32, f16, i16 | Done |
| `tsub` | `ub.vsub` | `VSUB.f32/f16/s16/s32` | f32, f16, i16 | Done |
| `tmul` | `ub.vmul` | `VMUL.f32/f16/s16/s32` | f32, f16, i16 | Done |
| `tdiv` | `ub.vdiv` | `VDIV.f32/f16/s16/s32` | f32, f16, i16 | Done |
| `tmax` | `ub.vmax` | `VMAX.f32/f16/s16/s32` | f32, f16, i16 | Done |
| `tmin` | `ub.vmin` | `VMIN.f32/f16/s16/s32` | f32, f16, i16 | Done |
| `tand` | `ub.vand` | `VAND.s16/u16` | i16 | Done |
| `tor`  | `ub.vor`  | `VOR.s16/u16`  | i16 | Done |
| `txor` | decompose | `VOR+VAND+VNOT+VAND` | i16 | Done (De Morgan) |

### Fused Tile-Tile Ops

| PTO Op | UB Op | Intrinsic | dtypes | Status |
|---|---|---|---|---|
| `taddrelu` | `ub.vaddrelu` | `VADDRELU.f32/f16/s16` | f32, f16, i16 | Done |

### Unary Ops (direct intrinsics)

| PTO Op | UB Op | Intrinsic | dtypes | Status |
|---|---|---|---|---|
| `tabs`  | `ub.vabs`  | `VABS.f32/f16/s16` | f32, f16 | Done |
| `trelu` | `ub.vrelu` | `VRELU.f32/f16/s32` | f32, f16 | Done |
| `texp`  | `ub.vexp`  | `VEXP.f32/f16` | f32, f16 | Done |
| `tlog`  | `ub.vln`   | `VLN.f32/f16` | f32, f16 | Done |
| `tsqrt` | `ub.vsqrt` | `VSQRT.f32/f16` | f32, f16 | Done |
| `trsqrt`| `ub.vrsqrt`| `VRSQRT.f32/f16` | f32, f16 | Done (fast-approx) |
| `tnot`  | `ub.vnot`  | `VNOT.s16/u16` | i16 | Done |

### Unary Ops (decomposed)

| PTO Op | Decomposition | Intrinsic(s) | dtypes | Status |
|---|---|---|---|---|
| `tneg` | `dst = src * (-1)` | `VMULS` | f32, f16 | Done |
| `trecip` | `dst = 1; dst = dst / src` | `MOVEV` + `VDIV` | f32, f16 | Done |

### Scalar-Tile Binary Ops

| PTO Op | UB Op | Intrinsic | dtypes | Status |
|---|---|---|---|---|
| `tadds` | `ub.vadds` | `VADDS.f32/f16/s16/s32` | f32, f16 | Done |
| `tmuls` | `ub.vmuls` | `VMULS.f32/f16/s16/s32` | f32, f16 | Done |
| `tmaxs` | `ub.vmaxs` | `VMAXS.f32/f16/s16/s32` | f32, f16 | Done |
| `tmins` | `ub.vmins` | `VMINS.f32/f16/s16/s32` | f32, f16 | Done |

### Scalar Shift Ops

| PTO Op | UB Op | Intrinsic | dtypes | Status |
|---|---|---|---|---|
| `tshls` | `ub.vshl` | `VSHL.u16` | i16 | Done |
| `tshrs` | `ub.vshr` | `VSHR.u16` | i16 | Done |

## Config Word Layouts

### Binary Tile-Tile Config (VADD/VSUB/VMUL/VDIV/VMAX/VMIN/VAND/VOR/VADDRELU)

```
 7:0   repeat
15:8   dst block stride
23:16  src0 block stride
31:24  src1 block stride
39:32  dst repeat stride
47:40  src0 repeat stride
55:48  src1 repeat stride
63:56  simd flag (=1)
```

### Unary Config (VABS/VRELU/VNOT/VEXP/VLN/VSQRT/VRSQRT)

```
15:0   dst block stride
31:16  src block stride
39:32  dst repeat stride
51:40  src repeat stride
63:56  repeat
```

### Scalar-Tile Config (VADDS/VMULS/VMAXS/VMINS/VSHL/VSHR/MOVEV)

Same as unary config layout (repeat at `[63:56]`).

## Scalar Bitcast Convention

Scalar-tile ops (`VADDS`/`VMULS`/`VMAXS`/`VMINS`) accept a scalar operand whose
LLVM-level type depends on the element type:

- **Float** (f32/f16): intrinsic takes `float`/`half` scalar
- **Integer** (s16/s32): intrinsic takes `i64` scalar (widened from int16/int32)

Since all UB ops store the scalar as `i64` in the PTO IR, float scalars require
bitcast-based conversion:

**Tile-to-UB lowering** (float scalar → i64):
```
f32 3.5 → bitcast → i32 → extsi → i64
f16 3.5 → bitcast → i16 → extsi → i64
```

**LLVM emitter** (i64 → float scalar):
```
i64 → trunc → i32 → bitcast → f32
i64 → trunc → i16 → bitcast → f16
```

This preserves the exact bit pattern, unlike `SIToFPOp` which would convert
the integer value (3 → 3.0, losing fractional parts).

For integer types (s16/s32), the i64 is passed directly to the intrinsic.

`tneg` must also use this bitcast-preserving path. Passing raw integer `i64 -1`
to `VMULS.f32/f16` truncates to all-one float bit patterns, which become NaN.
The lowering therefore materializes typed `-1.0` (`f32`/`f16`) and bitcasts it to
the UB scalar `i64` representation.

## Count Mode Limitation (C220)

C220 count mode (`SBITSET1(ctrl,56)` + `MOVEMASK(count)`) does **not** support
`repeat > 1`. Multi-element count mode causes CCE errors.

As a result:
- `repeat=1`: direct count mode (no loop)
- `repeat>1, rows=1`: split into per-chunk `scf.for` with repeat=1 (`modeCount1L`)
- `repeat>1, rows>1`: per-row `scf.for` with repeat=1 (`modeNorm1L`)

Non-VL-aligned shapes (e.g., 1×96, 1×200) are routed through `modeNorm1L`
(loop + tail), not `modeCount1L`.

## VRSQRT Fast Approximation

`VRSQRT` uses a hardware fast inverse square root approximation. The maximum
relative error is approximately 0.3%. E2e tests use `rtol=1e-2, atol=1e-2` for
rsqrt.

## TXOR Decomposition (No Native VXOR)

C220 has no native `VXOR` intrinsic. `TXOR` decomposes via De Morgan's law:

```
src0 ^ src1 = ~(src0 & src1) & (src0 | src1)
```

Lowered to:
```
VOR(tmp, src0, src1)         // tmp = src0 | src1
barrier PIPE_V
VAND(dst, src0, src1)        // dst = src0 & src1
barrier PIPE_V
VNOT(dst, dst)               // dst = ~dst
barrier PIPE_V
VAND(dst, dst, tmp)          // dst = dst & tmp
```

`PIPE_V` barriers between each step mirror CANN's `TXOR_IMPL` semantics.

## UB Memory Planning

A3 VPTO reuses the existing `PTOPlanMemory` pass:
```
PTOViewToMemref → PTOPlanMemory → PTOResolveReservedBuffers → PTOMaterializeTileHandles → LowerPTOToUBufOps
```

`LowerPTOToUBufOps` requires planned `alloc_tile addr = ...` from
`PTOMaterializeTileHandles`. The old manual sequential allocator has been
removed entirely.

See `docs/designs/a2a3-allocator.md` for details.

## Test Counts

- **60 UB lit tests** (tile-to-UB IR checks + UB-to-LLVM checks + round-trip + planned-address)
- **192 binary e2e** (f32/f16/i16 binary + bitwise + shift across dispatch-shape matrix, including `taddrelu`)
- **80 unary e2e** (abs/relu/neg/exp/log/sqrt/rsqrt/recip across 5 shapes × 2 dtypes)
- **120 scalar binary e2e** (adds/muls/maxs/mins across 5 shapes × 3 scalars × 2 dtypes)
- **Total confirmed elementwise hardware e2e tests: 392**

The PTODSL hardware e2e suite runs the A3 target by default. A2 and A3 share
the VPTO lowering pipeline, so this validates the A2/A3 lowering path unless
future lowering behavior diverges between the targets.
