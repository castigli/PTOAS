Qwen3 tilelet PTO kernels generated from `pypto-lib/examples/models/qwen3/qwen3_32b_decode_tilelet.py`.

Scope:
- compile-regression inputs for `ptoas`
- A5-only kernels; `runop.sh` injects `--pto-arch a5 --pto-level=level3` for this directory unless the caller already overrides `PTOAS_FLAGS`

Notes:
- The source PyPTO program lowers to a full orchestration file plus 5 ptoas-facing mixed-kernel `.pto` inputs:
  `qwen3_decode_layer_incore_1`, `qwen3_decode_layer_incore_2`,
  `qwen3_decode_layer_incore_10`, `qwen3_decode_layer_incore_13`,
  `qwen3_decode_layer_incore_14`.
- This sample directory vendors only those direct `ptoas` regression inputs.
- No custom `golden.py` or `compare.py` is included here: these grouped mixed kernels depend on orchestration-managed peer buffers and loop-carried context, so per-kernel numerical validation is not a drop-in replacement for the full PyPTO runtime flow.
