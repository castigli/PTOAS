# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TQuant INT8_SYM kernel sample.

  tquant(src_f32, scale_f32[row]) -> dst_i8

Loads a 32x32 f32 tile (src) and a 32x1 per-row scaling tile (scale),
performs symmetric INT8 quantization, and stores the int8 result tile.

Note: int8 tiles require Cols*sizeof(T) to be a multiple of 32 bytes
(the NPU aligned-size). At 1 byte/element that means Cols >= 32.
"""

from mlir.ir import (
    Attribute,
    Context,
    Location,
    Module,
    InsertionPoint,
    F32Type,
    IndexType,
    IntegerType,
    UnitAttr,
)
from mlir.dialects import func, arith, pto


# Tile shape used throughout the sample.
# int8/uint8 tiles require Cols*sizeof(T) % 32 == 0; use 32 cols minimum.
_SHAPE = [32, 32]


def _make_common_types(ctx):
    """Return a namespace of commonly used types / attrs."""
    f32 = F32Type.get(ctx)
    i8 = IntegerType.get_signless(8, ctx)
    idx = IndexType.get(ctx)

    ptr_f32 = pto.PtrType.get(f32, ctx)
    ptr_i8 = pto.PtrType.get(i8, ctx)

    tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
    tv2_i8 = pto.TensorViewType.get(2, i8, ctx)

    ptv_f32 = pto.PartitionTensorViewType.get(_SHAPE, f32, ctx)
    ptv_scale = pto.PartitionTensorViewType.get([_SHAPE[0], 1], f32, ctx)
    ptv_i8 = pto.PartitionTensorViewType.get(_SHAPE, i8, ctx)

    vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
    bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
    bl_col = pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx)
    sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
    pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
    cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
    cfg_col = pto.TileBufConfigAttr.get(
        bl_col, sl, pto.TileConfig.fractalABSize, pd, ctx
    )

    tb_f32 = pto.TileBufType.get(_SHAPE, f32, vec, _SHAPE, cfg, ctx)
    tb_scale = pto.TileBufType.get(
        [_SHAPE[0], 1], f32, vec, [_SHAPE[0], 1], cfg_col, ctx
    )
    tb_i8 = pto.TileBufType.get(_SHAPE, i8, vec, _SHAPE, cfg, ctx)

    quant_sym = pto.QuantTypeAttr.get(pto.QuantType.INT8_SYM, ctx)
    layout_dn = pto.LayoutAttr.get(pto.Layout.DN, ctx)

    class NS:
        pass

    ns = NS()
    ns.f32 = f32
    ns.i8 = i8
    ns.idx = idx
    ns.ptr_f32 = ptr_f32
    ns.ptr_i8 = ptr_i8
    ns.tv2_f32 = tv2_f32
    ns.tv2_i8 = tv2_i8
    ns.ptv_f32 = ptv_f32
    ns.ptv_scale = ptv_scale
    ns.ptv_i8 = ptv_i8
    ns.tb_f32 = tb_f32
    ns.tb_scale = tb_scale
    ns.tb_i8 = tb_i8
    ns.quant_sym = quant_sym
    ns.layout_dn = layout_dn
    return ns


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()
            t = _make_common_types(ctx)

            # ------------------------------------------------------------------
            # @tquant_sym_kernel(src_ptr: !pto.ptr<f32>,
            #                    scale_ptr: !pto.ptr<f32>,
            #                    dst_ptr: !pto.ptr<i8>)
            # ------------------------------------------------------------------
            fn_sym_ty = func.FunctionType.get([t.ptr_f32, t.ptr_f32, t.ptr_i8], [])
            with InsertionPoint(m.body):
                fn_sym = func.FuncOp("tquant_sym_kernel", fn_sym_ty)
                fn_sym.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                fn_sym.operation.attributes["pto.kernel_kind"] = Attribute.parse(
                    "#pto.kernel_kind<vector>", ctx
                )
                entry_sym = fn_sym.add_entry_block()

            with InsertionPoint(entry_sym):
                idx = t.idx
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c32 = arith.ConstantOp(idx, 32).result

                src_ptr, scale_ptr, dst_ptr = entry_sym.arguments

                # Make tensor views over the flat global-memory pointers.
                tv_src = pto.MakeTensorViewOp(
                    t.tv2_f32, src_ptr, [c32, c32], [c32, c1]
                ).result
                tv_scale = pto.MakeTensorViewOp(
                    t.tv2_f32,
                    scale_ptr,
                    [c32, c1],
                    [c1, c1],
                    layout=t.layout_dn,
                ).result
                tv_dst = pto.MakeTensorViewOp(
                    t.tv2_i8, dst_ptr, [c32, c32], [c32, c1]
                ).result

                # Partition into tile-sized sub-views.
                sv_src = pto.PartitionViewOp(
                    t.ptv_f32, tv_src, offsets=[c0, c0], sizes=[c32, c32]
                ).result
                sv_scale = pto.PartitionViewOp(
                    t.ptv_scale, tv_scale, offsets=[c0, c0], sizes=[c32, c1]
                ).result
                sv_dst = pto.PartitionViewOp(
                    t.ptv_i8, tv_dst, offsets=[c0, c0], sizes=[c32, c32]
                ).result

                # Allocate on-chip tile buffers.
                tb_src = pto.AllocTileOp(t.tb_f32).result
                tb_scale = pto.AllocTileOp(t.tb_scale).result
                tb_dst = pto.AllocTileOp(t.tb_i8).result

                # Load src and per-row scale tiles from global memory.
                pto.TLoadOp(None, sv_src, tb_src)
                pto.TLoadOp(None, sv_scale, tb_scale)

                # INT8_SYM quantization (no offset operand).
                pto.TQuantOp(tb_src, tb_scale, tb_dst, quant_type=t.quant_sym)

                # Store result back to global memory.
                pto.TStoreOp(None, tb_dst, sv_dst)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
