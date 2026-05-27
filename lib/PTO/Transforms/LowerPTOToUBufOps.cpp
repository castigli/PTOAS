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
//   6. Total element count aligned to elementsPerRepeat
//
// CCE lowering formula (from TAdd.hpp):
//   elementsPerRepeat = 128 / sizeof(T)
//   blockSizeElem     = 32  / sizeof(T)
//   repeat            = (vRows * vCols) / elementsPerRepeat
//   repeatStride      = cols / blockSizeElem
//
// Emitted op: pto.ub.vadd dst, src0, src1, repeat, 1,1,1, repStride, repStride, repStride
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

static bool canLower(pto::TAddOp op) {
  auto dstTy = dyn_cast<pto::TileBufType>(op.getDst().getType());
  auto src0Ty = dyn_cast<pto::TileBufType>(op.getSrc0().getType());
  auto src1Ty = dyn_cast<pto::TileBufType>(op.getSrc1().getType());
  if (!dstTy || !src0Ty || !src1Ty)
    return false;

  if (!isUBMemorySpace(dstTy) || !isUBMemorySpace(src0Ty) ||
      !isUBMemorySpace(src1Ty))
    return false;
  if (!isRowMajor(dstTy) || !isRowMajor(src0Ty) || !isRowMajor(src1Ty))
    return false;

  Type elemTy = dstTy.getElementType();
  unsigned elemSize = getElementSize(elemTy);
  if (elemSize == 0)
    return false;

  auto shape = dstTy.getShape();
  auto validShape = dstTy.getValidShape();
  if (shape.size() < 2)
    return false;

  int64_t vRows = (!validShape.empty() &&
                   validShape[0] != ShapedType::kDynamic)
                      ? validShape[0]
                      : shape[0];
  int64_t vCols = (validShape.size() >= 2 &&
                   validShape[1] != ShapedType::kDynamic)
                      ? validShape[1]
                      : shape[1];
  if (vRows == ShapedType::kDynamic || vCols == ShapedType::kDynamic)
    return false;

  unsigned elementsPerRepeat = 128 / elemSize;
  if ((vRows * vCols) % elementsPerRepeat != 0)
    return false;

  return true;
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

      builder.setInsertionPoint(op);
      lowerTAdd(op, builder);
      op.erase();
    }
  }

private:
  void lowerTAdd(pto::TAddOp op, OpBuilder &builder) {
    auto dstTy = cast<pto::TileBufType>(op.getDst().getType());
    auto src0Ty = cast<pto::TileBufType>(op.getSrc0().getType());
    auto src1Ty = cast<pto::TileBufType>(op.getSrc1().getType());
    Type elemTy = dstTy.getElementType();
    MLIRContext *ctx = builder.getContext();
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

    Type elemTy0 = src0Ty.getElementType();
    Type elemTy1 = src1Ty.getElementType();
    (void)elemTy0;
    (void)elemTy1;

    unsigned elemSize = getElementSize(elemTy);
    unsigned elementsPerRepeat = 128 / elemSize;
    unsigned blockSizeElem = 32 / elemSize;

    auto shape = dstTy.getShape();
    auto validShape = dstTy.getValidShape();
    int64_t vRows = (!validShape.empty() &&
                     validShape[0] != ShapedType::kDynamic)
                        ? validShape[0]
                        : shape[0];
    int64_t vCols = (validShape.size() >= 2 &&
                     validShape[1] != ShapedType::kDynamic)
                        ? validShape[1]
                        : shape[1];
    int64_t cols = shape[1];

    int64_t repeat = (vRows * vCols) / static_cast<int64_t>(elementsPerRepeat);
    int64_t repeatStride = cols / static_cast<int64_t>(blockSizeElem);

    auto getI64 = [&](int64_t val) -> Value {
      return builder.create<arith::ConstantOp>(
          loc, builder.getI64IntegerAttr(val));
    };

    Value cOne = getI64(1);
    Value cRepeat = getI64(repeat);
    Value cRepStride = getI64(repeatStride);

    builder.create<pto::UBVaddOp>(loc, dstPtr, src0Ptr, src1Ptr, cRepeat, cOne,
                                  cOne, cOne, cRepStride, cRepStride,
                                  cRepStride);
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
