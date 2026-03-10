# PyPTO IR Parser Fixed PTO Cases

These cases are vendored snapshots of `.pto` files generated from `pypto/examples/ir_parser`
on `main` so PTOAS does not depend on PyPTO's future behavior at test time.

Source provenance:
- PyPTO repo: `https://github.com/hw-native-sys/pypto`
- PyPTO commit: `9f6985f7b92acb1754cf479e179427afcdd1934a`
- Source directory: `examples/ir_parser`

Generation settings used to materialize these files:
- `backend_type=BackendType.PTO`
- `strategy=OptimizationStrategy.PTOAS`
- `skip_ptoas=True`

Included positive cases:
- `orchestration_example`: `kernel_add`, `kernel_add_scalar`, `kernel_mul`
- `vector_example_dag`: `kernel_add`, `kernel_add_scalar`, `kernel_mul`
- `paged_attention_example`: `kernel_init_inplace`, `kernel_qk_matmul`, `kernel_softmax_prepare`, `kernel_pv_matmul`, `kernel_online_update`

Intentionally not included:
- `program_example`: parser/program roundtrip example, does not emit PTO kernels
- `batch_paged_attention_example`: currently fails on PyPTO `main`
- `paged_attention_with_incore`: currently fails on PyPTO `main`
- `paged_attention_tensor`: currently fails on PyPTO `main`
- `flash_attention_parsing`: currently fails on PyPTO `main`

If upstream PyPTO changes these examples, update these snapshots manually after re-validating
that the new `.pto` still compiles with PTOAS.
