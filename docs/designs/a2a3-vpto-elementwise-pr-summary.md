# A2/A3 VPTO Elementwise PR Summary

## Summary

This branch adds the A2/A3 VPTO UB lowering path for tile elementwise operations and validates it end-to-end on A3 hardware. The implementation routes A2/A3 through the planned-memory UB pipeline instead of the old/manual allocation path, lowers supported PTO tile ops to VPTO UB ops, and emits CANN900 LLVM intrinsics.

## What Changed

- Added the A2/A3 VPTO UB lowering pipeline:
  `PTOViewToMemref -> PTOPlanMemory -> PTOResolveReservedBuffers -> PTOMaterializeTileHandles -> LowerPTOToUBufOps`.
- Implemented A2/A3 UB lowering for binary elementwise ops:
  `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin`, `tand`, `tor`, `txor`, `tshls`, and `tshrs`.
- Implemented unary elementwise ops:
  `tabs`, `trelu`, `tneg`, `texp`, `tlog`, `tsqrt`, `trsqrt`, and `trecip`.
- Implemented scalar-tile elementwise ops:
  `tadds`, `tmuls`, `tmaxs`, and `tmins`.
- Added fused elementwise support for `taddrelu`, including PTO IR, UB IR, lowering, LLVM intrinsic emission, PTODSL wrappers, and tests.
- Added UB/LLVM support for new VPTO UB ops and intrinsics, including `vaddrelu`, `vdup`, and `vln`.
- Integrated planned UB address allocation so UB lowering consumes planner-assigned `alloc_tile addr` values and models scratch usage for ops such as `txor`.
- Extended dispatch coverage for small, normal, count, row-repeat, tail, chunked, and multi-row shapes.
- Extended the PTODSL elementwise surface through helpers and `pto.tile.*` APIs.
- Added lit and hardware e2e coverage for binary, unary, scalar, bitwise, shift, fused, log, and reciprocal cases.

## Validation

Verified after merging current `main` with the LLVM 21 docker image:

- Build: passed
- VPTO UB lit: `61/61`
- Runtime toolchain pytest: `4/4`
- A3 unary hardware e2e: `80/80`
- A3 binary hardware e2e: `192/192`
- A3 scalar hardware e2e: `120/120`

Total A2/A3 shared VPTO elementwise hardware coverage validated on A3: `392` tests.
