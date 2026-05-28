// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- LowerPTOToUBufOps.cpp - Lower pto.tadd to pto.ub.vadd on a2a3 -----===//
//===----------------------------------------------------------------------===//
//
// Lowers pto.tadd to pto.ub.vadd on a3 (dav-m200-vec). Uses CCE-derived
// formulas for repeat count and stride computation from tile shape metadata.
//
// Eligibility gates:
//   1. Target arch is a3
//   2. Tiles are in UB memory space
//   3. Row-major layout
//   4. Supported element type (f16, f32, i16, i32)
//   5. Shapes are compile-time static
//   6. Full-width valid region (vCols == cols) or single row (vRows == 1)
//
// CCE lowering formula (from TAdd.hpp):
//   elementsPerRepeat = 128 / sizeof(T)
//   blockSizeElem     = 32  / sizeof(T)
//   repeat            = (vRows * vCols) / elementsPerRepeat
//   repeatStride      = cols / blockSizeElem
//
// Lowering modes (emulating CCE strategies):
//   Single-chunk (repeat <= 255):
//     pto.ub.vadd dst, src0, src1, repeat, 1,1,1, repStride×3
//
//   Chunked (repeat > 255):
//     scf.for loop over 255-repeat chunks + tail, mirroring Bin2LNormModeHead.
//
//   Masked tail (repeat % elementsPerRepeat != 0):
//     pto.ub.set_mask tailMask + pto.ub.vadd(1) for tail elements,
//     then pto.ub.set_mask -1 to restore full mask.
//
//   No barriers needed — all vadd ops execute in PIPE_V and are
//   inherently ordered.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace pto {
  #define GEN_PASS_DEF_LOWERPTOTOUBUFOPS
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

static constexpr int64_t kRepeatMax = 255;
static constexpr unsigned kMaskLen = 64;

static unsigned getElementSize(Type elemTy) {
  if (elemTy.isF16() || elemTy.isBF16())
    return 2;
  if (elemTy.isF32())
    return 4;
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    unsigned width = intTy.getWidth();
    if (width == 16 || width == 32)
      return width / 8;
  }
  return 0;
}

static bool isUBMemorySpace(pto::TileBufType tbTy) {
  auto msAttr =
      dyn_cast_or_null<pto::AddressSpaceAttr>(tbTy.getMemorySpace());
  if (!msAttr)
    return false;
  auto space = msAttr.getAddressSpace();
  return space == pto::AddressSpace::VEC ||
         space == pto::AddressSpace::SCALING ||
         space == pto::AddressSpace::Zero;
}

static bool isRowMajor(pto::TileBufType tbTy) {
  auto config = tbTy.getConfigAttr();
  if (!config)
    return true;
  return config.getBLayout().getValue() != pto::BLayout::ColMajor;
}

static pto::PtrType getUBPtrType(MLIRContext *ctx, Type elemTy) {
  auto msAttr =
      pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC);
  return pto::PtrType::get(ctx, elemTy, msAttr);
}

static std::pair<int64_t, int64_t>
computeContMaskValues(unsigned nElements) {
  int64_t mask0 = (nElements >= kMaskLen)
      ? static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)
      : static_cast<int64_t>((1ULL << nElements) - 1ULL);
  int64_t mask1 = (nElements > kMaskLen)
      ? static_cast<int64_t>((1ULL << (nElements - kMaskLen)) - 1ULL)
      : 0LL;
  return {mask0, mask1};
}

struct TileShapeInfo {
  int64_t vRows;
  int64_t vCols;
  int64_t cols;
  unsigned elemSize;
  unsigned elementsPerRepeat;
  unsigned blockSizeElem;
};

static std::optional<TileShapeInfo> extractTileShapeInfo(pto::TAddOp op) {
  auto dstTy = dyn_cast<pto::TileBufType>(op.getDst().getType());
  auto src0Ty = dyn_cast<pto::TileBufType>(op.getSrc0().getType());
  auto src1Ty = dyn_cast<pto::TileBufType>(op.getSrc1().getType());
  if (!dstTy || !src0Ty || !src1Ty)
    return std::nullopt;

  if (!isUBMemorySpace(dstTy) || !isUBMemorySpace(src0Ty) ||
      !isUBMemorySpace(src1Ty))
    return std::nullopt;
  if (!isRowMajor(dstTy) || !isRowMajor(src0Ty) || !isRowMajor(src1Ty))
    return std::nullopt;

  Type elemTy = dstTy.getElementType();
  unsigned elemSize = getElementSize(elemTy);
  if (elemSize == 0)
    return std::nullopt;

  auto shape = dstTy.getShape();
  auto validShape = dstTy.getValidShape();
  if (shape.size() < 2)
    return std::nullopt;

  int64_t vRows = (!validShape.empty() &&
                   validShape[0] != ShapedType::kDynamic)
                      ? validShape[0]
                      : shape[0];
  int64_t vCols = (validShape.size() >= 2 &&
                   validShape[1] != ShapedType::kDynamic)
                      ? validShape[1]
                      : shape[1];
  if (vRows == ShapedType::kDynamic || vCols == ShapedType::kDynamic)
    return std::nullopt;

  TileShapeInfo info;
  info.vRows = vRows;
  info.vCols = vCols;
  info.cols = shape[1];
  info.elemSize = elemSize;
  info.elementsPerRepeat = 128 / elemSize;
  info.blockSizeElem = 32 / elemSize;

  // Flat lowering requires valid region to be full-width or single-row.
  // Partial valid columns need row-based lowering (deferred to v3).
  if (vCols != info.cols && vRows != 1)
    return std::nullopt;

  return info;
}

static bool canLower(pto::TAddOp op) {
  return extractTileShapeInfo(op).has_value();
}

struct LowerPTOToUBufOpsPass
    : public pto::impl::LowerPTOToUBufOpsBase<LowerPTOToUBufOpsPass> {
  using LowerPTOToUBufOpsBase::LowerPTOToUBufOpsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    auto mod = func->getParentOfType<ModuleOp>();
    if (!mod)
      return;
    auto archAttr = mod->getAttrOfType<StringAttr>("pto.target_arch");
    if (!archAttr || archAttr.getValue() != "a3")
      return;

    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    SmallVector<pto::TAddOp> taddOps;
    func.walk([&](pto::TAddOp op) { taddOps.push_back(op); });

    for (auto op : taddOps) {
      if (!canLower(op))
        continue;

      auto info = extractTileShapeInfo(op);
      if (!info)
        continue;

      builder.setInsertionPoint(op);

      Type elemTy = cast<pto::TileBufType>(op.getDst().getType())
                        .getElementType();
      auto ptrType = getUBPtrType(ctx, elemTy);
      Location loc = op.getLoc();

      auto emitAddr = [&](Value tile) -> Value {
        auto addrOp =
            builder.create<pto::TileBufAddrOp>(loc, ptrType, tile);
        return addrOp.getDst();
      };

      Value dstPtr = emitAddr(op.getDst());
      Value src0Ptr = emitAddr(op.getSrc0());
      Value src1Ptr = emitAddr(op.getSrc1());

      int64_t totalV = info->vRows * info->vCols;
      int64_t headRepeats =
          totalV / static_cast<int64_t>(info->elementsPerRepeat);
      int64_t tailElements =
          totalV % static_cast<int64_t>(info->elementsPerRepeat);
      int64_t repStride =
          info->cols / static_cast<int64_t>(info->blockSizeElem);

      if (headRepeats > 0) {
        if (headRepeats <= kRepeatMax)
          lowerTAddSingle(loc, builder, dstPtr, src0Ptr, src1Ptr, headRepeats,
                          repStride);
        else
          lowerTAddChunked(loc, builder, dstPtr, src0Ptr, src1Ptr, headRepeats,
                           repStride, info->elementsPerRepeat);
      }

      if (tailElements > 0)
        lowerTAddMaskedTail(loc, builder, dstPtr, src0Ptr, src1Ptr,
                            headRepeats, tailElements, repStride,
                            info->elementsPerRepeat);

      op.erase();
    }
  }

private:
  void lowerTAddSingle(Location loc, OpBuilder &builder, Value dstPtr,
                       Value src0Ptr, Value src1Ptr, int64_t repeat,
                       int64_t repStride) {
    auto getI64 = [&](int64_t val) -> Value {
      return builder.create<arith::ConstantOp>(
          loc, builder.getI64IntegerAttr(val));
    };

    Value cOne = getI64(1);
    Value cRepeat = getI64(repeat);
    Value cRepStride = getI64(repStride);

    builder.create<pto::UBVaddOp>(loc, dstPtr, src0Ptr, src1Ptr, cRepeat, cOne,
                                  cOne, cOne, cRepStride, cRepStride,
                                  cRepStride);
  }

  void lowerTAddChunked(Location loc, OpBuilder &builder, Value dstPtr,
                        Value src0Ptr, Value src1Ptr, int64_t repeat,
                        int64_t repStride, unsigned elementsPerRepeat) {
    int64_t numChunks = repeat / kRepeatMax;
    int64_t tailRepeats = repeat % kRepeatMax;
    int64_t elementsPerChunk =
        static_cast<int64_t>(kRepeatMax) * elementsPerRepeat;

    auto getI64 = [&](int64_t val) -> Value {
      return builder.create<arith::ConstantOp>(
          loc, builder.getI64IntegerAttr(val));
    };
    auto getIndex = [&](int64_t val) -> Value {
      return builder
          .create<arith::ConstantOp>(
              loc, builder.getIntegerAttr(builder.getIndexType(), val))
          .getResult();
    };

    Value c0 = getIndex(0);
    Value c1 = getIndex(1);
    Value cNumChunks = getIndex(numChunks);
    Value cElemPerChunk = getIndex(elementsPerChunk);

    Value c255I64 = getI64(kRepeatMax);
    Value c1I64 = getI64(1);
    Value cRepStride = getI64(repStride);

    auto forOp = builder.create<scf::ForOp>(loc, c0, cNumChunks, c1);
    builder.setInsertionPointToStart(forOp.getBody());
    Value i = forOp.getInductionVar();

    Value elemOff =
        builder.create<arith::MulIOp>(loc, i, cElemPerChunk).getResult();

    Value chunkDst =
        builder.create<pto::AddPtrOp>(loc, dstPtr.getType(), dstPtr, elemOff);
    Value chunkSrc0 = builder.create<pto::AddPtrOp>(loc, src0Ptr.getType(),
                                                    src0Ptr, elemOff);
    Value chunkSrc1 = builder.create<pto::AddPtrOp>(loc, src1Ptr.getType(),
                                                    src1Ptr, elemOff);

    builder.create<pto::UBVaddOp>(loc, chunkDst, chunkSrc0, chunkSrc1, c255I64,
                                  c1I64, c1I64, c1I64, cRepStride, cRepStride,
                                  cRepStride);

    builder.setInsertionPointAfter(forOp);

    if (tailRepeats > 0) {
      Value cTailRepeats = getI64(tailRepeats);
      Value tailOffset = getIndex(numChunks * elementsPerChunk);

      Value tailDst =
          builder.create<pto::AddPtrOp>(loc, dstPtr.getType(), dstPtr,
                                        tailOffset);
      Value tailSrc0 = builder.create<pto::AddPtrOp>(
          loc, src0Ptr.getType(), src0Ptr, tailOffset);
      Value tailSrc1 = builder.create<pto::AddPtrOp>(
          loc, src1Ptr.getType(), src1Ptr, tailOffset);

      builder.create<pto::UBVaddOp>(loc, tailDst, tailSrc0, tailSrc1,
                                    cTailRepeats, c1I64, c1I64, c1I64,
                                    cRepStride, cRepStride, cRepStride);
    }
  }

  void lowerTAddMaskedTail(Location loc, OpBuilder &builder, Value dstPtr,
                           Value src0Ptr, Value src1Ptr, int64_t headRepeats,
                           unsigned tailElements, int64_t repStride,
                           unsigned elementsPerRepeat) {
    auto [mask0Val, mask1Val] = computeContMaskValues(tailElements);

    auto getI64 = [&](int64_t val) -> Value {
      return builder.create<arith::ConstantOp>(
          loc, builder.getI64IntegerAttr(val));
    };
    auto getIndex = [&](int64_t val) -> Value {
      return builder
          .create<arith::ConstantOp>(
              loc, builder.getIntegerAttr(builder.getIndexType(), val))
          .getResult();
    };

    Value m0 = getI64(mask0Val);
    Value m1 = getI64(mask1Val);
    builder.create<pto::UBSetMaskOp>(loc, m0, m1);

    int64_t tailOffset = headRepeats * elementsPerRepeat;
    Value off = getIndex(tailOffset);

    Value td =
        builder.create<pto::AddPtrOp>(loc, dstPtr.getType(), dstPtr, off);
    Value ts0 =
        builder.create<pto::AddPtrOp>(loc, src0Ptr.getType(), src0Ptr, off);
    Value ts1 =
        builder.create<pto::AddPtrOp>(loc, src1Ptr.getType(), src1Ptr, off);

    Value c1 = getI64(1);
    Value cRS = getI64(repStride);

    builder.create<pto::UBVaddOp>(loc, td, ts0, ts1, c1, c1, c1, c1, cRS, cRS,
                                  cRS);

    Value cFull = getI64(-1);
    builder.create<pto::UBSetMaskOp>(loc, cFull, cFull);
  }
};

} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createLowerPTOToUBufOpsPass() {
  return std::make_unique<LowerPTOToUBufOpsPass>();
}
} // namespace pto
} // namespace mlir
