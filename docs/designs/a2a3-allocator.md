# A2/A3 VPTO UB Allocator Design

## Context

The A3 VPTO path lowers tile-level operations directly to UB pointer intrinsics
such as `pto.ub.vadd`, `pto.ub.vand`, `pto.ub.vor`, and `pto.ub.vnot`.

The current implementation in `LowerPTOToUBufOps` assigns UB addresses manually
while replacing `pto.alloc_tile` with `pto.castptr` constants. This was enough
for early binary elementwise work, but it diverges from the existing memory
planner used by the non-A3 path.

A recent `TXOR` issue exposed the weakness of this approach:

- `TXOR` needs a scratch `tmp` tile.
- Manual allocation used overly large spacing.
- The fourth tile could be placed at or beyond A3 UB capacity.
- The fix reduced spacing and added capacity checks, but this is still a local
  allocator.

Long term, A3 should reuse `PTOPlanMemory`.

## Current A3 Path

```text
A3 VPTO:
  LowerPTOToUBufOps
    alloc_tile -> constant UB pointer
    tile op -> pto.ub.* op
  return from lowerPTOToVPTOBackend
```

Characteristics:

- Pointer-based.
- No memref lowering before UB intrinsic lowering.
- No `PTOPlanMemory`.
- No liveness reuse.
- Manual capacity/alignment policy in `LowerPTOToUBufOps`.

## Current A5 / Non-A3 Path

```text
A5 / non-A3:
  PTOViewToMemref
  PTOPlanMemory
  PTOResolveReservedBuffers
  PTOMaterializeTileHandles
  ExpandTileOp
  PTOInlineLibCall
  FoldTileBufIntrinsics
```

Characteristics:

- Tile buffers become memref-backed local buffers.
- `PTOPlanMemory` assigns local memory.
- Planned addresses are later materialized back into tile handles.
- EmitC consumes planned tile addresses through `TASSIGN(tile, addr)`.

This is the model A3 should converge toward.

## Option 1: Reuse Existing A5-Style Planning Path

Pipeline target:

```text
A3:
  PTOViewToMemref
  PTOPlanMemory
  PTOResolveReservedBuffers
  PTOMaterializeTileHandles
  LowerPTOToUBufOps
  VPTO pointer/LLVM lowering
```

### Meaning

Convert A3 tile allocations into memref-backed local buffers before UB intrinsic
lowering. Then run the existing `PTOPlanMemory` pass to assign UB addresses.
After planning, re-materialize tile handles and let `LowerPTOToUBufOps` consume
planned addresses instead of assigning offsets itself.

### Why This Works

`PTOPlanMemory` already understands:

- local VEC capacity
- 256-byte alignment
- liveness
- buffer reuse
- DPS init buffers
- scratch buffers via memory effects

`TXorOp::getEffects()` already models `tmp` as a scratch write on A2/A3, so
`tmp` can participate in planning once it is represented as a local buffer.

### Pros

- Reuses the existing allocator directly.
- Matches A5 / EmitC architecture.
- Gets liveness reuse and capacity diagnostics.
- Avoids a second allocator policy in `LowerPTOToUBufOps`.

### Cons

- Requires A3 pipeline refactor.
- Requires validating that A3 UB lowering works after tile-to-memref-to-tile
  materialization.
- May require adjusting `LowerPTOToUBufOps` to reject unplanned tiles or consume
  planned `addr` values.

### Recommended

This is the preferred option.

## Option 2: Extend Planner Logic To Tile Buffers Directly

Pipeline target:

```text
A3:
  A3TileAddressPlanning
  LowerPTOToUBufOps
```

### Meaning

Keep A3 tile-native. Teach planner logic to understand `pto.alloc_tile`
directly, or extract reusable allocation/liveness code from `PTOPlanMemory` and
apply it to tile buffers before lowering.

### Pros

- Less disruption to current A3 VPTO path.
- Avoids memref round trip.
- Keeps tile-level IR intact until UB lowering.

### Cons

- More planner work.
- Higher risk of duplicating `PTOPlanMemory`.
- Easier to diverge from A5 allocation semantics.
- Requires new planner surface for tile buffers.

### Not Preferred Initially

Use only if Option 1 reveals hard incompatibilities with the
memref/materialized-tile seam.

## Conservative Migration Plan

### Step 1: Keep Current Fix As Safety Net

Keep the temporary manual allocator in `LowerPTOToUBufOps` with:

- 256-byte alignment
- A3 UB capacity checks
- clear diagnostics

This remains as a fallback during migration.

### Step 2: Add A3 Pre-Planning Pipeline

Enable an A3 path equivalent to:

```text
PTOViewToMemref
PTOPlanMemory
PTOResolveReservedBuffers
PTOMaterializeTileHandles
```

before A3 UB intrinsic lowering.

### Step 3: Split `LowerPTOToUBufOps`

Separate responsibilities:

```text
LowerPTOToUBufAddressFallback
  only temporary/manual allocation

LowerPTOToUBufOps
  consume planned tile addresses
  lower tile ops to pto.ub.* / MTE ops
```

The long-term lowering pass should not invent UB addresses.

### Step 4: Remove Manual Allocator

Once lit and hardware e2e pass with planned addresses:

- remove manual sequential allocation
- require planned addresses for A3 UB lowering
- keep explicit diagnostics for missing planning

## Validation Plan

Required tests:

- UB lit suite
- binary elementwise hardware e2e
- `TXOR` hardware e2e with scratch tile
- overflow test proving capacity diagnostics
- planning test showing `tmp` is planned and not placed out of bounds

Expected command examples:

```bash
ninja -C build ptoas
/opt/llvm-project/build/bin/llvm-lit build/test/lit --filter 'vpto/ub'
PYTHONPATH=/workspace/ptodsl python3 -m pytest ptodsl/tests/e2e/test_binary_elementwise.py -v
```

## Decision

Proceed with Option 1.

A3 should reuse the existing `PTOPlanMemory` path by converting tile buffers to
memref-backed local buffers before UB intrinsic lowering, then materializing
planned tile handles for `LowerPTOToUBufOps`.

## Implementation Status

Option 1 is implemented and validated for A3 VPTO:

- A3 VPTO enters the local tile planning path.
- `PTOViewToMemref`, `PTOPlanMemory`, `PTOResolveReservedBuffers`, and
  `PTOMaterializeTileHandles` run before A3 UB intrinsic lowering.
- `LowerPTOToUBufOps` requires planned addresses unconditionally; the manual
  sequential allocator has been removed entirely.
- A3 transfer lowering accepts planned-path GM memref views and lowers them to
  MTE ops before VPTO LLVM emission.
- Dead `memref.subview` / `memref.reinterpret_cast` / `memref.cast` ops are
  cleaned up after transfer lowering.
- Lit test `vpto/ub/planned_addresses/txor_planned_addr.pto` verifies that all
  four TXOR tiles (including `tmp`) receive compact planned addresses.

### Migration Step Progress

| Step | Status |
|---|---|
| Step 1: Keep current fix as safety net | Superseded — manual allocator removed entirely |
| Step 2: Add A3 pre-planning pipeline | Done — `PTOViewToMemref` + `PTOPlanMemory` + materialization |
| Step 3: Split `LowerPTOToUBufOps` | N/A — manual fallback removed, no split needed |
| Step 4: Remove manual allocator | Done — `LowerPTOToUBufOps` requires planned addresses unconditionally |

### Validated Coverage

- UB lit suite: `49 passed` (48 existing + 1 planned-address test).
