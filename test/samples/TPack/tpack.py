# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, InsertionPoint, Location, Module, IndexType, IntegerType, StringAttr
from mlir.dialects import arith, func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()
            module.operation.attributes["pto.target_arch"] = StringAttr.get("a5")

            i16 = IntegerType.get_signless(16, ctx)
            i32 = IntegerType.get_signless(32, ctx)
            idx = IndexType.get(ctx)
            ptr_i32 = pto.PtrType.get(i32, ctx)
            ptr_i16 = pto.PtrType.get(i16, ctx)
            tv2_i32 = pto.TensorViewType.get(2, i32, ctx)
            tv2_i16 = pto.TensorViewType.get(2, i16, ctx)
            ptv_128x128_i32 = pto.PartitionTensorViewType.get([128, 128], i32, ctx)
            ptv_128x128_i16 = pto.PartitionTensorViewType.get([128, 128], i16, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            src_tb = pto.TileBufType.get([128, 128], i32, vec, [128, 128], cfg, ctx)
            dst_tb = pto.TileBufType.get([128, 128], i16, vec, [128, 128], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_i32, ptr_i16], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("tpack_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c128 = arith.ConstantOp(idx, 128).result

                src_view = pto.MakeTensorViewOp(tv2_i32, entry.arguments[0], [c128, c128], [c128, c1]).result
                dst_view = pto.MakeTensorViewOp(tv2_i16, entry.arguments[1], [c128, c128], [c128, c1]).result
                src_part = pto.PartitionViewOp(
                    ptv_128x128_i32, src_view, offsets=[c0, c0], sizes=[c128, c128]
                ).result
                dst_part = pto.PartitionViewOp(
                    ptv_128x128_i16, dst_view, offsets=[c0, c0], sizes=[c128, c128]
                ).result

                src_tile = pto.AllocTileOp(src_tb).result
                dst_tile = pto.AllocTileOp(dst_tb).result

                pto.TLoadOp(None, src_part, src_tile)
                pto.TPackOp(src_tile, dst_tile)
                pto.TStoreOp(None, dst_tile, dst_part)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
