# A2/A3 VPTO EmitC Parity Roadmap

## Goal

Close the A2/A3 VPTO vector-operation gap relative to the EmitC backend without
guessing hardware contracts. Each implementation slice must fail closed outside
its documented safe subset and remain independently reviewable and promotable.

Development branches and stacked pull requests live in `castigli/PTOAS` until
they are ready to be promoted upstream one at a time.

Tracking issues:

- [A2/A3 VPTO parity roadmap](https://github.com/castigli/PTOAS/issues/4)
- [F1: reject residual tile operations](https://github.com/castigli/PTOAS/issues/5)

## Branch Structure

The current gather work is the root of several shallow thematic stacks:

```text
feature/a2a3-mgather-vpto
└── feature/a2a3-vpto-fail-closed
    ├── quick-wins stack
    ├── reductions stack
    ├── compare/select stack
    ├── conversion stack
    └── scatter stack
```

The first pull request in each thematic stack targets
`feature/a2a3-vpto-fail-closed`. Later pull requests target the preceding branch
in that stack. Hardware-contract probes are tracked as blocking issues and do
not get placeholder implementation branches.

## Foundation

| ID | Branch | Scope |
| --- | --- | --- |
| F1 | `feature/a2a3-vpto-fail-closed` | Reject residual `TileOpInterface` operations before VPTO emission and make both LLVM emitter conversion targets reject them as a backstop. |

F1 is a prerequisite for every later stack. Unsupported types, shapes, modes,
or operations must produce a targeted diagnostic instead of surviving into
LLVM emission.

## Quick Wins

| ID | Branch | Scope | Base |
| --- | --- | --- | --- |
| Q1 | `feature/a2a3-tsubs-tlrelu-vpto` | Lower `TSubS` through `VADDS` with a negated scalar and add the proven `TLRelu` lowering. | F1 |
| Q2 | `feature/a2a3-treshape-texpands-vpto` | Implement metadata-only `TReshape` and lower `TExpands` through `pto.ub.vdup`. | Q1 |

This stack can proceed while the hardware probes for larger operation families
run.

## Reductions And Expands

| ID | Branch | Scope | Base |
| --- | --- | --- | --- |
| R0 | Issue only | Establish C220 contracts for `VCOPY`, `VBRCB`, `VCADD`, `VCMAX`, and `VCMIN`. | N/A |
| R1 | `feature/a2a3-ub-reduce-broadcast-vpto` | Add the verified raw UB reduction and broadcast substrate. | F1 |
| R2 | `feature/a2a3-rowcol-reduce-vpto` | Lower row/column sum, maximum, and minimum. | R1 |
| R3 | `feature/a2a3-rowcol-expand-vpto` | Lower row/column expand, add, multiply, subtract, maximum, and minimum. | R2 |
| R4 | `feature/a2a3-argreduce-prod-vpto` | Lower argmax, argmin, and product after their contracts are proven. | R3 |

Division and `Expdif` expansions remain deferred until operand order, edge
cases, and backend intrinsic behavior are demonstrated.

## Compare And Select

| ID | Branch | Scope | Base |
| --- | --- | --- | --- |
| C0 | Issue only | Probe packed-mask layout, predicates, `SET.CMPMASK`, `VSEL`, temporary-tile behavior, tails, and aliases on A2 and A3. | N/A |
| C1 | `feature/a2a3-ub-compare-vpto` | Add raw vector-vector and vector-scalar compare operations, initially `EQ` for `f16`, `f32`, and `i32`. | F1 |
| C2 | `feature/a2a3-tcmp-eq-vpto` | Lower `TCmp` and `TCmpS` for the proven `EQ` subset. | C1 |
| C3 | `feature/a2a3-ub-select-vpto` | Represent compare-mask state and add the raw select substrate. | C2 |
| C4 | `feature/a2a3-tsel-vpto` | Lower `TSel` for proven `f16` and `f32` layouts. | C3 |
| C5 | `feature/a2a3-tsels-vpto` | Lower `TSelS` while preserving the established temporary-tile contract. | C4 |

Additional predicates and integer select forms require follow-up issues rather
than broadening the initial subset without evidence.

## Conversions

| ID | Branch | Scope | Base |
| --- | --- | --- | --- |
| V0 | Issue only | Probe `VCONV` strides, tails, rounding suffixes, saturation controls, and alias restrictions. | N/A |
| V1 | `feature/a2a3-ub-vconv-vpto` | Add the minimal verified raw `VCONV` substrate. | F1 |
| V2 | `feature/a2a3-tcvt-i16-f32-vpto` | Lower `TCvt<i16, f32>` with `CAST_RINT` and saturation disabled. | V1 |
| V3 | `feature/a2a3-tdequant-i16-vpto` | Decompose i16 dequantization into conversion, offset, and scale operations. | V2 |

Quantization, i8 dequantization, and broader conversion pairs remain separate
contract issues.

## Scatter

| ID | Branch | Scope | Base |
| --- | --- | --- | --- |
| S0 | Issue only | Probe raw mask-scatter ABI, addressing, aliases, and synchronization. | N/A |
| S1 | `feature/a2a3-tscatter-vpto` | Implement static ND VEC indexed `TScatter` with a safe scalar UB loop. | F1 |
| S2 | `feature/a2a3-mscatter-vpto` | Implement explicit-coalesce, non-atomic Row/Elem `MScatter` for statically safe GM metadata. | S1 |

Mask scatter, MAT/L1 paths, FIX/cube paths, and atomic behavior stay deferred.

## Deferred Families

Create tracking issues, but no implementation branches, until the necessary
hardware contracts are available:

- `TMov`, `TTrans`, `TExtract`, `TInsert`, and `TFillPad*`
- `TCI`, `TTri`, and `TPrefetch`
- `TFMod`, `TRem`, and `TPow`
- Sort and partial reductions
- Quantization and i8 dequantization

## Acceptance Criteria

Every implementation pull request must include the applicable parts of this
test matrix:

- Exact tile-to-UB lowering tests
- Exact UB-to-LLVM intrinsic and configuration tests
- Negative tests for every explicitly deferred type, mode, shape, and layout
- Proof that no supported high-level tile operation remains after lowering
- Planned-address, padded-row, full-repeat, and tail coverage
- PTODSL wrapper tests when a public operation is added
- A3 numerical tests with source and destination guard regions
- C220 evidence before claiming A2 support
- Full VPTO lit validation without new A5 regressions

## Promotion Workflow

1. Open each pull request in `castigli/PTOAS` against its stack predecessor.
2. Keep a pull request in draft until focused tests pass and its predecessor is
   stable.
3. Promote F1 upstream first.
4. After each upstream merge, rebase the next branch onto upstream `main`, rerun
   validation, and open the corresponding upstream pull request.
5. Preserve unsupported cases as explicit diagnostics until a later issue adds
   evidence and coverage for them.
