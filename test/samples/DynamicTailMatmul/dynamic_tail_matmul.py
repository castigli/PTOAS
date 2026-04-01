# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint, IndexType, IntegerType, F16Type, F32Type
from mlir.dialects import func, arith, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            f32 = F32Type.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)

            ptr_a = pto.PtrType.get(f16, ctx)
            ptr_b = pto.PtrType.get(f16, ctx)
            ptr_c = pto.PtrType.get(f32, ctx)

            tv2_a = pto.TensorViewType.get(2, f16, ctx)
            tv2_b = pto.TensorViewType.get(2, f16, ctx)
            tv2_c = pto.TensorViewType.get(2, f32, ctx)

            view_a = pto.PartitionTensorViewType.get([32, 32], f16, ctx)
            view_b = pto.PartitionTensorViewType.get([32, 32], f16, ctx)
            view_c = pto.PartitionTensorViewType.get([32, 32], f32, ctx)

            cfg_mat = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pto.PadValueAttr.get(pto.PadValue.Null, ctx),
                ctx,
            )
            cfg_left = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pto.PadValueAttr.get(pto.PadValue.Null, ctx),
                ctx,
            )
            cfg_right = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.ColMajor, ctx),
                pto.TileConfig.fractalABSize,
                pto.PadValueAttr.get(pto.PadValue.Null, ctx),
                ctx,
            )
            cfg_acc = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalCSize,
                pto.PadValueAttr.get(pto.PadValue.Null, ctx),
                ctx,
            )

            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT, ctx)
            right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT, ctx)
            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)

            # Dynamic valid shape for tail blocks: m/k/n come from runtime arguments.
            a_mat_ty = pto.TileBufType.get([32, 32], f16, mat, [-1, -1], cfg_mat, ctx)
            b_mat_ty = pto.TileBufType.get([32, 32], f16, mat, [-1, -1], cfg_mat, ctx)
            a_left_ty = pto.TileBufType.get([32, 32], f16, left, [-1, -1], cfg_left, ctx)
            b_right_ty = pto.TileBufType.get([32, 32], f16, right, [-1, -1], cfg_right, ctx)
            c_acc_ty = pto.TileBufType.get([32, 32], f32, acc, [-1, -1], cfg_acc, ctx)

            # (A, B, C, validM, validK, validN)
            fn_ty = func.FunctionType.get([ptr_a, ptr_b, ptr_c, i32, i32, i32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("dynamic_tail_matmul_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                a_ptr, b_ptr, c_ptr, vm_i32, vk_i32, vn_i32 = entry.arguments

                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                vm = arith.IndexCastOp(IndexType.get(ctx), vm_i32).result
                vk = arith.IndexCastOp(IndexType.get(ctx), vk_i32).result
                vn = arith.IndexCastOp(IndexType.get(ctx), vn_i32).result

                tv_a = pto.MakeTensorViewOp(tv2_a, a_ptr, [c32, c32], [c32, c1]).result
                tv_b = pto.MakeTensorViewOp(tv2_b, b_ptr, [c32, c32], [c32, c1]).result
                tv_c = pto.MakeTensorViewOp(tv2_c, c_ptr, [c32, c32], [c32, c1]).result

                sv_a = pto.PartitionViewOp(view_a, tv_a, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_b = pto.PartitionViewOp(view_b, tv_b, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_c = pto.PartitionViewOp(view_c, tv_c, offsets=[c0, c0], sizes=[c32, c32]).result

                a_mat = pto.AllocTileOp(a_mat_ty, valid_row=vm, valid_col=vk).result
                b_mat = pto.AllocTileOp(b_mat_ty, valid_row=vk, valid_col=vn).result
                a_left = pto.AllocTileOp(a_left_ty, valid_row=vm, valid_col=vk).result
                b_right = pto.AllocTileOp(b_right_ty, valid_row=vk, valid_col=vn).result
                c_acc = pto.AllocTileOp(c_acc_ty, valid_row=vm, valid_col=vn).result

                pto.TLoadOp(None, sv_a, a_mat)
                pto.TLoadOp(None, sv_b, b_mat)
                pto.TMovOp(None, a_mat, a_left)
                pto.TMovOp(None, b_mat, b_right)
                pto.TMatmulOp(None, a_left, b_right, c_acc)
                pto.TStoreOp(None, c_acc, sv_c)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
