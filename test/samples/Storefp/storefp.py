from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, pto
from mlir.ir import F32Type, IntegerType, MemRefType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            i8 = IntegerType.get_signless(8, ctx)
            u64 = IntegerType.get_unsigned(64, ctx)

            gm = pto.AddressSpaceAttr.get(pto.AddressSpace.GM, ctx)
            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)
            scaling = pto.AddressSpaceAttr.get(pto.AddressSpace.SCALING, ctx)

            pad = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg_acc = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalCSize,
                pad,
                ctx,
            )
            cfg_fp = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pad,
                ctx,
            )

            acc_tile_ty = pto.TileBufType.get([16, 32], f32, acc, [1, 32], cfg_acc, ctx)
            fp_tile_ty = pto.TileBufType.get([1, 16], u64, scaling, [1, 16], cfg_fp, ctx)
            dst_memref_ty = MemRefType.get([1, 32], i8, memory_space=gm)

            fn_ty = func.FunctionType.get([dst_memref_ty], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tstore_fp_pass", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                dst = entry.arguments[0]
                acc_tile = pto.AllocTileOp(acc_tile_ty).result
                fp_tile = pto.AllocTileOp(fp_tile_ty).result
                pto.TStoreFPOp(acc_tile, fp_tile, dst)
                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
