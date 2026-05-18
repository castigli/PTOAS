#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Basic smoke tests for pto_ssa dialect Python bindings."""

from mlir.ir import Context, Module
from mlir.dialects import pto


def assert_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {needle!r} in\n{text}")


def main() -> None:
    with Context() as ctx:
        # Register both dialects
        pto.register_dialect(ctx)
        assert hasattr(pto, "register_ssa_dialect"), (
            "register_ssa_dialect not exported from pto module"
        )
        pto.register_ssa_dialect(ctx)

        # Parse a minimal pto_ssa module and round-trip it
        asm = r"""
module {
  func.func @vadd_ssa(
      %a : !pto.partition_tensor_view<32x32xf16>,
      %b : !pto.partition_tensor_view<32x32xf16>,
      %out : !pto.partition_tensor_view<32x32xf16>) {
    %ta = pto_ssa.tload ins(%a : !pto.partition_tensor_view<32x32xf16>)
          : !pto.tile_buf<[32, 32], f16, VEC>
    %tb = pto_ssa.tload ins(%b : !pto.partition_tensor_view<32x32xf16>)
          : !pto.tile_buf<[32, 32], f16, VEC>
    %tc = pto_ssa.tvadd ins(%ta : !pto.tile_buf<[32, 32], f16, VEC>,
                            %tb : !pto.tile_buf<[32, 32], f16, VEC>)
          : !pto.tile_buf<[32, 32], f16, VEC>
    pto_ssa.tstore ins(%tc : !pto.tile_buf<[32, 32], f16, VEC>)
                  outs(%out : !pto.partition_tensor_view<32x32xf16>)
    return
  }
}
"""
        mod = Module.parse(asm, ctx)
        printed = str(mod)
        assert_contains(printed, "pto_ssa.tload")
        assert_contains(printed, "pto_ssa.tvadd")
        assert_contains(printed, "pto_ssa.tstore")
        print("PASS: ssa_ops.py")


if __name__ == "__main__":
    main()
