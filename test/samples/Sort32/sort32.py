from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IntegerType, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)

            u32 = IntegerType.get_unsigned(32, ctx)
            ptr_u32 = pto.PtrType.get(u32, ctx)
            tv2_u32 = pto.TensorViewType.get(2, u32, ctx)

            tile_view_f32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            tile_view_u32 = pto.PartitionTensorViewType.get([32, 32], u32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_f32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)
            tile_buf_u32 = pto.TileBufType.get([32, 32], u32, vec, [32, 32], cfg, ctx)

            def build_sort32_func(name: str, use_tmp: bool):
                fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_u32], [])
                with InsertionPoint(m.body):
                    fn = func.FuncOp(name, fn_ty)
                    entry = fn.add_entry_block()

                with InsertionPoint(entry):
                    c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                    c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                    c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                    arg0, arg1, arg2 = entry.arguments

                    tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result
                    tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result
                    tv2 = pto.MakeTensorViewOp(tv2_u32, arg2, [c32, c32], [c32, c1]).result

                    sv0 = pto.PartitionViewOp(tile_view_f32, tv0, offsets=[c0, c0], sizes=[c32, c32]).result
                    sv1 = pto.PartitionViewOp(tile_view_f32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result
                    sv2 = pto.PartitionViewOp(tile_view_u32, tv2, offsets=[c0, c0], sizes=[c32, c32]).result

                    tb0 = pto.AllocTileOp(tile_buf_f32).result
                    tb1 = pto.AllocTileOp(tile_buf_f32).result
                    tb2 = pto.AllocTileOp(tile_buf_u32).result

                    pto.TLoadOp(None, sv0, tb0)
                    pto.TLoadOp(None, sv2, tb2)

                    if use_tmp:
                        tb_tmp = pto.AllocTileOp(tile_buf_f32).result
                        pto.TSort32Op(tb0, tb2, tb1, tmp=tb_tmp)
                    else:
                        pto.TSort32Op(tb0, tb2, tb1)

                    pto.TStoreOp(None, tb1, sv1)
                    pto.TStoreOp(None, tb2, sv2)
                    func.ReturnOp([])

            build_sort32_func("sort32_kernel_2d", use_tmp=False)
            build_sort32_func("sort32_kernel_2d_tmp", use_tmp=True)

            m.operation.verify()

            return m


if __name__ == "__main__":
    print(build())
