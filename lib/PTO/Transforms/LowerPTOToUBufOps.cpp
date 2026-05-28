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
// Lowers pto.tadd to pto.ub.vadd on a3 (dav-m200-vec). Uses the full CCE
// dispatch tree from TBinOp.hpp with all modes.
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
static constexpr int64_t kRepeatStrideMax = 255;
static constexpr int64_t kSmallRptBinOp = 4;
static constexpr int64_t kDefaultRepeatStride = 8;
static constexpr unsigned kMaskLen = 64;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

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
  auto msAttr = pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC);
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
  int64_t rows;
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

  int64_t rows = shape[0];
  int64_t cols = shape[1];
  int64_t vRows = (!validShape.empty() &&
                   validShape[0] != ShapedType::kDynamic)
                      ? validShape[0] : rows;
  int64_t vCols = (validShape.size() >= 2 &&
                   validShape[1] != ShapedType::kDynamic)
                      ? validShape[1] : cols;
  if (vRows == ShapedType::kDynamic || vCols == ShapedType::kDynamic ||
      rows == ShapedType::kDynamic || cols == ShapedType::kDynamic)
    return std::nullopt;

  TileShapeInfo info;
  info.vRows = vRows;
  info.vCols = vCols;
  info.cols = cols;
  info.rows = rows;
  info.elemSize = elemSize;
  info.elementsPerRepeat = 128 / elemSize;
  info.blockSizeElem = 32 / elemSize;
  return info;
}

static bool canLower(pto::TAddOp op) {
  return extractTileShapeInfo(op).has_value();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

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
        auto addrOp = builder.create<pto::TileBufAddrOp>(loc, ptrType, tile);
        return addrOp.getDst();
      };

      Value dstPtr = emitAddr(op.getDst());
      Value src0Ptr = emitAddr(op.getSrc0());
      Value src1Ptr = emitAddr(op.getSrc1());
      dispatch(loc, builder, dstPtr, src0Ptr, src1Ptr, ptrType, *info);
      op.erase();
    }
  }

private:
  //===--------------------------------------------------------------------===//
  // Helpers
  //===--------------------------------------------------------------------===//

  Value i64c(int64_t val, Location loc, OpBuilder &b) {
    return b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(val));
  }
  Value idxc(int64_t val, Location loc, OpBuilder &b) {
    return b.create<arith::ConstantOp>(
               loc, b.getIntegerAttr(b.getIndexType(), val))
        .getResult();
  }
  Value i64c0(Location loc, OpBuilder &b) { return i64c(0, loc, b); }
  Value i64c1(Location loc, OpBuilder &b) { return i64c(1, loc, b); }
  Value i64cM1(Location loc, OpBuilder &b) { return i64c(-1, loc, b); }
  Value i64c8(Location loc, OpBuilder &b) { return i64c(kDefaultRepeatStride, loc, b); }
  Value idxc0(Location loc, OpBuilder &b) { return idxc(0, loc, b); }
  Value idxc1(Location loc, OpBuilder &b) { return idxc(1, loc, b); }

  void vadd(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
            Value repeat, Value repStride) {
    b.create<pto::UBVaddOp>(loc, dst, s0, s1, repeat,
                            i64c1(loc, b), i64c1(loc, b), i64c1(loc, b),
                            repStride, repStride, repStride);
  }

  void setMask(Location loc, OpBuilder &b, unsigned n) {
    auto [m0, m1] = computeContMaskValues(n);
    b.create<pto::UBSetMaskOp>(loc, i64c(m0, loc, b), i64c(m1, loc, b));
  }

  void fullMask(Location loc, OpBuilder &b) {
    b.create<pto::UBSetMaskOp>(loc, i64cM1(loc, b), i64cM1(loc, b));
  }

  Value addPtr(Location loc, OpBuilder &b, Value base, pto::PtrType ptrTy,
               Value off) {
    return b.create<pto::AddPtrOp>(loc, ptrTy, base, off);
  }

  //===--------------------------------------------------------------------===//
  // CCE dispatch tree — mirrors TBinOp.hpp BinaryInstr
  //===--------------------------------------------------------------------===//

  void dispatch(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t epr = info.elementsPerRepeat;
    int64_t cols = info.cols;
    int64_t rows = info.rows;
    int64_t vRows = info.vRows;
    int64_t vCols = info.vCols;

    // 1. Small tile
    if (rows <= kRepeatMax && cols < static_cast<int64_t>(epr)) {
      modeSmall(loc, b, dst, s0, s1, ptrTy, info);
      return;
    }

    // 2. Continuous at compile time
    if (vCols == cols || vRows == 1) {
      int64_t totalV = vRows * vCols;
      int64_t totalRpts = (totalV + epr - 1) / epr;
      bool nonVLAligned =
          (vCols > static_cast<int64_t>(epr)) && ((vCols % epr) != 0);

      if (nonVLAligned || totalRpts > kRepeatMax)
        modeCount1L(loc, b, dst, s0, s1, ptrTy, info);
      else
        modeNorm1L(loc, b, dst, s0, s1, ptrTy, info);
      return;
    }

    // 3. Non-continuous
    int64_t normColRepeat = cols / epr;
    if (normColRepeat > 1 && vRows * normColRepeat < kSmallRptBinOp) {
      modeCount2L(loc, b, dst, s0, s1, ptrTy, info);
    } else if (vRows < normColRepeat + 1) {
      if (vCols % epr > 0)
        modeCount2L(loc, b, dst, s0, s1, ptrTy, info);
      else
        modeColVLAlign(loc, b, dst, s0, s1, ptrTy, info);
    } else {
      modeRowRpt(loc, b, dst, s0, s1, ptrTy, info);
    }
  }

  //===--------------------------------------------------------------------===//
  // Bin1LNormModeSmall
  //===--------------------------------------------------------------------===//

  void modeSmall(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                 pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t rs = info.cols / static_cast<int64_t>(info.blockSizeElem);
    setMask(loc, b, info.vCols);
    vadd(loc, b, dst, s0, s1, i64c(info.vRows, loc, b), i64c(rs, loc, b));
    fullMask(loc, b);
  }

  //===--------------------------------------------------------------------===//
  // Bin1LNormMode – flat, stride=8, repeat≤255
  //===--------------------------------------------------------------------===//

  void modeNorm1L(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                  pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t totalV = info.vRows * info.vCols;
    int64_t epr = info.elementsPerRepeat;
    int64_t headRepeats = totalV / epr;
    int64_t tailElements = totalV % epr;

    if (headRepeats > 0)
      vadd(loc, b, dst, s0, s1, i64c(headRepeats, loc, b), i64c8(loc, b));

    if (tailElements > 0) {
      Value off = idxc(headRepeats * epr, loc, b);
      Value td = addPtr(loc, b, dst, ptrTy, off);
      Value ts0 = addPtr(loc, b, s0, ptrTy, off);
      Value ts1 = addPtr(loc, b, s1, ptrTy, off);
      setMask(loc, b, tailElements);
      vadd(loc, b, td, ts0, ts1, i64c1(loc, b), i64c8(loc, b));
      fullMask(loc, b);
    }
  }

  //===--------------------------------------------------------------------===//
  // Bin1LCountMode
  //===--------------------------------------------------------------------===//

  void modeCount1L(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                   pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t totalV = info.vRows * info.vCols;
    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(totalV, loc, b), i64c0(loc, b));
    vadd(loc, b, dst, s0, s1, i64c0(loc, b), i64c8(loc, b));
    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeColVLAlign
  //===--------------------------------------------------------------------===//

  void modeColVLAlign(Location loc, OpBuilder &b, Value dst, Value s0,
                      Value s1, pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t epr = info.elementsPerRepeat;
    int64_t headRepeats = info.vCols / epr;
    int64_t rowStride = info.cols;

    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(info.vRows, loc, b), idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value off = b.create<arith::MulIOp>(loc, iv, idxc(rowStride, loc, b))
                    .getResult();
    Value rd = addPtr(loc, b, dst, ptrTy, off);
    Value rs0 = addPtr(loc, b, s0, ptrTy, off);
    Value rs1 = addPtr(loc, b, s1, ptrTy, off);
    vadd(loc, b, rd, rs0, rs1, i64c(headRepeats, loc, b), i64c8(loc, b));
    b.setInsertionPointAfter(forOp);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LCountMode – row-by-row count mode
  //===--------------------------------------------------------------------===//

  void modeCount2L(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                   pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t rowStride = info.cols;
    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(info.vCols, loc, b),
                               i64c0(loc, b));

    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(info.vRows, loc, b), idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value off = b.create<arith::MulIOp>(loc, iv, idxc(rowStride, loc, b))
                    .getResult();
    Value rd = addPtr(loc, b, dst, ptrTy, off);
    Value rs0 = addPtr(loc, b, s0, ptrTy, off);
    Value rs1 = addPtr(loc, b, s1, ptrTy, off);
    vadd(loc, b, rd, rs0, rs1, i64c0(loc, b), i64c8(loc, b));
    b.setInsertionPointAfter(forOp);

    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeRowRpt
  //===--------------------------------------------------------------------===//

  void modeRowRpt(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                   pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t be = info.blockSizeElem;
    int64_t rowStride = info.cols;
    int64_t rs = rowStride / be;
    bool condRowRpt = (info.vRows <= kRepeatMax) && (rs <= kRepeatStrideMax);

    if (condRowRpt)
      rowRptFast(loc, b, dst, s0, s1, ptrTy, info, rs);
    else
      rowRptChunked(loc, b, dst, s0, s1, ptrTy, info, rowStride, rs);
  }

  void rowRptFast(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                  pto::PtrType ptrTy, const TileShapeInfo &info, int64_t rs) {
    int64_t epr = info.elementsPerRepeat;
    int64_t numLoop = info.vCols / epr;
    int64_t tailElements = info.vCols % epr;

    for (int64_t i = 0; i < numLoop; i++) {
      Value rd = addPtr(loc, b, dst, ptrTy, idxc(i * epr, loc, b));
      Value r0 = addPtr(loc, b, s0, ptrTy, idxc(i * epr, loc, b));
      Value r1 = addPtr(loc, b, s1, ptrTy, idxc(i * epr, loc, b));
      vadd(loc, b, rd, r0, r1, i64c(info.vRows, loc, b), i64c(rs, loc, b));
    }

    if (tailElements > 0) {
      Value off = idxc(numLoop * epr, loc, b);
      Value rd = addPtr(loc, b, dst, ptrTy, off);
      Value r0 = addPtr(loc, b, s0, ptrTy, off);
      Value r1 = addPtr(loc, b, s1, ptrTy, off);
      setMask(loc, b, tailElements);
      vadd(loc, b, rd, r0, r1, i64c(info.vRows, loc, b), i64c(rs, loc, b));
      fullMask(loc, b);
    }
  }

  void rowRptChunked(Location loc, OpBuilder &b, Value dst, Value s0,
                     Value s1, pto::PtrType ptrTy, const TileShapeInfo &info,
                     int64_t rowStride, int64_t rs) {
    int64_t epr = info.elementsPerRepeat;
    int64_t rptPerLine = info.vCols / epr;
    int64_t remainElem = info.vCols % epr;

    if (info.vRows > static_cast<int64_t>(epr)) {
      if (rptPerLine > 0)
        headRows(loc, b, dst, s0, s1, ptrTy, info, rowStride, rptPerLine);
      if (remainElem > 0) {
        Value off = idxc(rptPerLine * epr, loc, b);
        tailRows(loc, b, addPtr(loc, b, dst, ptrTy, off),
                 addPtr(loc, b, s0, ptrTy, off),
                 addPtr(loc, b, s1, ptrTy, off), ptrTy, info, rowStride, rs,
                 remainElem);
      }
    } else {
      if (remainElem == 0) {
        headRows(loc, b, dst, s0, s1, ptrTy, info, rowStride,
                 info.vCols / epr);
      } else if (rptPerLine > 0) {
        headRows(loc, b, dst, s0, s1, ptrTy, info, rowStride, rptPerLine);
        Value off = idxc(rptPerLine * epr, loc, b);
        tailRows(loc, b, addPtr(loc, b, dst, ptrTy, off),
                 addPtr(loc, b, s0, ptrTy, off),
                 addPtr(loc, b, s1, ptrTy, off), ptrTy, info, rowStride, rs,
                 remainElem);
      } else {
        tailRows(loc, b, dst, s0, s1, ptrTy, info, rowStride, rs, remainElem);
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeHead – chunked per-row head
  //===--------------------------------------------------------------------===//

  void headRows(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                pto::PtrType ptrTy, const TileShapeInfo &info,
                int64_t rowStride, int64_t rptPerLine) {
    int64_t epr = info.elementsPerRepeat;
    int64_t numLoop = rptPerLine / kRepeatMax;
    int64_t remain = rptPerLine % kRepeatMax;
    int64_t chunkElems = kRepeatMax * epr;

    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(info.vRows, loc, b),
                                      idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value rowBase =
        b.create<arith::MulIOp>(loc, iv, idxc(rowStride, loc, b)).getResult();

    if (numLoop > 0) {
      auto inner = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                        idxc(numLoop, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(inner.getBody());
      Value jv = inner.getInductionVar();
      Value co = b.create<arith::MulIOp>(loc, jv, idxc(chunkElems, loc, b))
                     .getResult();
      Value off = b.create<arith::AddIOp>(loc, rowBase, co).getResult();
      vadd(loc, b, addPtr(loc, b, dst, ptrTy, off),
           addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
           i64c(kRepeatMax, loc, b), i64c8(loc, b));
      b.setInsertionPointAfter(inner);
    }

    if (remain > 0) {
      Value co = idxc(numLoop * chunkElems, loc, b);
      Value off = b.create<arith::AddIOp>(loc, rowBase, co).getResult();
      vadd(loc, b, addPtr(loc, b, dst, ptrTy, off),
           addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
           i64c(remain, loc, b), i64c8(loc, b));
    }
    b.setInsertionPointAfter(forOp);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeTail – masked per-row tail
  //===--------------------------------------------------------------------===//

  void tailRows(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                pto::PtrType ptrTy, const TileShapeInfo &info,
                int64_t rowStride, int64_t rs, unsigned remainPerLine) {
    bool strideOver =
        (rowStride / info.blockSizeElem > kRepeatStrideMax);
    setMask(loc, b, remainPerLine);

    int64_t numLoop = 0;
    int64_t remainAfterLoop = info.vRows;
    if (info.vRows > kRepeatMax) {
      numLoop = info.vRows / kRepeatMax;
      remainAfterLoop = info.vRows % kRepeatMax;

      auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                        idxc(numLoop, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      if (strideOver)
        tailStrideOverChunk(loc, b, iv, dst, s0, s1, ptrTy, rowStride);
      else
        tailStrideOkChunk(loc, b, iv, dst, s0, s1, ptrTy, rowStride, rs);
      b.setInsertionPointAfter(forOp);
    }

    if (remainAfterLoop > 0) {
      if (strideOver)
        tailStrideOverRemain(loc, b, dst, s0, s1, ptrTy, rowStride, numLoop,
                             remainAfterLoop);
      else
        tailStrideOkRemain(loc, b, dst, s0, s1, ptrTy, rowStride, rs, numLoop,
                           remainAfterLoop);
    }

    fullMask(loc, b);
  }

  void tailStrideOverChunk(Location loc, OpBuilder &b, Value iv, Value dst,
                           Value s0, Value s1, pto::PtrType ptrTy,
                           int64_t rowStride) {
    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(kRepeatMax, loc, b), idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value jv = forOp.getInductionVar();
    Value baseOff = b.create<arith::MulIOp>(
        loc, iv, idxc(kRepeatMax * rowStride, loc, b)).getResult();
    Value rowOff =
        b.create<arith::MulIOp>(loc, jv, idxc(rowStride, loc, b)).getResult();
    Value off = b.create<arith::AddIOp>(loc, baseOff, rowOff).getResult();
    vadd(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c1(loc, b), i64c1(loc, b));
    b.setInsertionPointAfter(forOp);
  }

  void tailStrideOkChunk(Location loc, OpBuilder &b, Value iv, Value dst,
                         Value s0, Value s1, pto::PtrType ptrTy,
                         int64_t rowStride, int64_t rs) {
    Value off = b.create<arith::MulIOp>(
        loc, iv, idxc(kRepeatMax * rowStride, loc, b)).getResult();
    vadd(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c(kRepeatMax, loc, b), i64c(rs, loc, b));
  }

  void tailStrideOverRemain(Location loc, OpBuilder &b, Value dst, Value s0,
                            Value s1, pto::PtrType ptrTy, int64_t rowStride,
                            int64_t numLoop, int64_t remain) {
    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b), idxc(remain, loc, b),
                                      idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value jv = forOp.getInductionVar();
    Value baseOff = idxc(numLoop * kRepeatMax * rowStride, loc, b);
    Value rowOff =
        b.create<arith::MulIOp>(loc, jv, idxc(rowStride, loc, b)).getResult();
    Value off = b.create<arith::AddIOp>(loc, baseOff, rowOff).getResult();
    vadd(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c1(loc, b), i64c1(loc, b));
    b.setInsertionPointAfter(forOp);
  }

  void tailStrideOkRemain(Location loc, OpBuilder &b, Value dst, Value s0,
                          Value s1, pto::PtrType ptrTy, int64_t rowStride,
                          int64_t rs, int64_t numLoop, int64_t remain) {
    Value off = idxc(numLoop * kRepeatMax * rowStride, loc, b);
    vadd(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c(remain, loc, b), i64c(rs, loc, b));
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
