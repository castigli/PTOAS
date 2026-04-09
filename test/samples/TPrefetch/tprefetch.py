# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, InsertionPoint, Location, Module, IndexType, F16Type
from mlir.dialects import arith, func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()
            f16 = F16Type.get(ctx)
            idx = IndexType.get(ctx)
            ptr_f16 = pto.PtrType.get(f16, ctx)
            tv2_f16 = pto.TensorViewType.get(2, f16, ctx)
            ptv_16x16 = pto.PartitionTensorViewType.get([16, 16], f16, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tb_16x16 = pto.TileBufType.get([16, 16], f16, vec, [16, 16], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f16, ptr_f16], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("tprefetch_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c16 = arith.ConstantOp(idx, 16).result

                src_view = pto.MakeTensorViewOp(tv2_f16, entry.arguments[0], [c16, c16], [c16, c1]).result
                dst_view = pto.MakeTensorViewOp(tv2_f16, entry.arguments[1], [c16, c16], [c16, c1]).result
                src_part = pto.PartitionViewOp(
                    ptv_16x16, src_view, offsets=[c0, c0], sizes=[c16, c16]
                ).result
                dst_part = pto.PartitionViewOp(
                    ptv_16x16, dst_view, offsets=[c0, c0], sizes=[c16, c16]
                ).result

                tile = pto.AllocTileOp(tb_16x16).result
                pto.TPrefetchOp(src_part, tile)
                pto.TStoreOp(None, tile, dst_part)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
