# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TDequant int8 kernel sample.

  tdequant(src_i8, scale_f32, offset_f32) -> dst_f32

Loads a 32x32 i8 tile (src), a 32x8 f32 per-row scale tile, and a 32x8 f32
per-row offset tile, applies dequantization, and stores the f32 result tile.

Formula: dst[i][j] = (float(src[i][j]) - offset[i][0]) * scale[i][0]

The i8→f32 path uses a two-step conversion (int8→half→float) on A2/A3 hardware
and a direct vectorized path on A5.

Note: i8 tiles require Cols*sizeof(T) to be a multiple of 32 bytes
(the NPU aligned-size). At 1 byte/element that means Cols >= 32.
f32 parameter tiles: Cols*4 >= 32 means Cols >= 8.
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


# Tile shapes
_SRC_SHAPE   = [32, 32]   # i8 source tile (32 cols × 1 byte = 32 bytes ✓)
_PARA_SHAPE  = [32, 8]    # f32 parameter tile (scale / offset)
_DST_SHAPE   = [32, 32]   # f32 output tile


def _make_common_types(ctx):
    """Return a namespace of commonly used types / attrs."""
    f32 = F32Type.get(ctx)
    i8  = IntegerType.get_signless(8, ctx)
    idx = IndexType.get(ctx)

    ptr_f32 = pto.PtrType.get(f32, ctx)
    ptr_i8  = pto.PtrType.get(i8,  ctx)

    tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
    tv2_i8  = pto.TensorViewType.get(2, i8,  ctx)

    ptv_i8_src   = pto.PartitionTensorViewType.get(_SRC_SHAPE,  i8,  ctx)
    ptv_f32_src  = pto.PartitionTensorViewType.get(_DST_SHAPE,  f32, ctx)
    ptv_f32_para = pto.PartitionTensorViewType.get(_PARA_SHAPE, f32, ctx)

    vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
    bl  = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
    sl  = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
    pd  = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
    cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)

    tb_i8       = pto.TileBufType.get(_SRC_SHAPE,  i8,  vec, _SRC_SHAPE,  cfg, ctx)
    tb_f32_dst  = pto.TileBufType.get(_DST_SHAPE,  f32, vec, _DST_SHAPE,  cfg, ctx)
    tb_f32_para = pto.TileBufType.get(_PARA_SHAPE, f32, vec, _PARA_SHAPE, cfg, ctx)

    class NS:
        pass

    ns = NS()
    ns.f32          = f32
    ns.i8           = i8
    ns.idx          = idx
    ns.ptr_f32      = ptr_f32
    ns.ptr_i8       = ptr_i8
    ns.tv2_f32      = tv2_f32
    ns.tv2_i8       = tv2_i8
    ns.ptv_i8_src   = ptv_i8_src
    ns.ptv_f32_src  = ptv_f32_src
    ns.ptv_f32_para = ptv_f32_para
    ns.tb_i8        = tb_i8
    ns.tb_f32_dst   = tb_f32_dst
    ns.tb_f32_para  = tb_f32_para
    return ns


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()
            t = _make_common_types(ctx)

            # ------------------------------------------------------------------
            # @tdequant_i8_kernel(src_ptr:    !pto.ptr<i8>,
            #                     scale_ptr:  !pto.ptr<f32>,
            #                     offset_ptr: !pto.ptr<f32>,
            #                     dst_ptr:    !pto.ptr<f32>)
            # ------------------------------------------------------------------
            fn_ty = func.FunctionType.get(
                [t.ptr_i8, t.ptr_f32, t.ptr_f32, t.ptr_f32], []
            )
            with InsertionPoint(m.body):
                fn = func.FuncOp("tdequant_i8_kernel", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                fn.operation.attributes["pto.kernel_kind"] = Attribute.parse(
                    "#pto.kernel_kind<vector>", ctx
                )
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                idx = t.idx
                c0  = arith.ConstantOp(idx, 0).result
                c1  = arith.ConstantOp(idx, 1).result
                c8  = arith.ConstantOp(idx, 8).result
                c32 = arith.ConstantOp(idx, 32).result

                src_ptr, scale_ptr, offset_ptr, dst_ptr = entry.arguments

                # Make tensor views over the flat global-memory pointers.
                tv_src    = pto.MakeTensorViewOp(
                    t.tv2_i8,  src_ptr,    [c32, c32], [c32, c1]
                ).result
                tv_scale  = pto.MakeTensorViewOp(
                    t.tv2_f32, scale_ptr,  [c32, c8],  [c8,  c1]
                ).result
                tv_offset = pto.MakeTensorViewOp(
                    t.tv2_f32, offset_ptr, [c32, c8],  [c8,  c1]
                ).result
                tv_dst    = pto.MakeTensorViewOp(
                    t.tv2_f32, dst_ptr,    [c32, c32], [c32, c1]
                ).result

                # Partition into tile-sized sub-views.
                sv_src    = pto.PartitionViewOp(
                    t.ptv_i8_src,   tv_src,    offsets=[c0, c0], sizes=[c32, c32]
                ).result
                sv_scale  = pto.PartitionViewOp(
                    t.ptv_f32_para, tv_scale,  offsets=[c0, c0], sizes=[c32, c8]
                ).result
                sv_offset = pto.PartitionViewOp(
                    t.ptv_f32_para, tv_offset, offsets=[c0, c0], sizes=[c32, c8]
                ).result
                sv_dst    = pto.PartitionViewOp(
                    t.ptv_f32_src,  tv_dst,    offsets=[c0, c0], sizes=[c32, c32]
                ).result

                # Allocate on-chip tile buffers.
                tb_src    = pto.AllocTileOp(t.tb_i8).result
                tb_scale  = pto.AllocTileOp(t.tb_f32_para).result
                tb_offset = pto.AllocTileOp(t.tb_f32_para).result
                tb_dst    = pto.AllocTileOp(t.tb_f32_dst).result

                # Load tiles from global memory.
                pto.TLoadOp(None, sv_src,    tb_src)
                pto.TLoadOp(None, sv_scale,  tb_scale)
                pto.TLoadOp(None, sv_offset, tb_offset)

                # Dequantize: dst[i][j] = (float(src[i][j]) - offset[i][0]) * scale[i][0]
                pto.TDequantOp(tb_src, tb_scale, tb_offset, tb_dst)

                # Store result back to global memory.
                pto.TStoreOp(None, tb_dst, sv_dst)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
