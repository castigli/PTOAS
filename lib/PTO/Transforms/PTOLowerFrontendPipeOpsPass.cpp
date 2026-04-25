// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOLOWERFRONTENDPIPEOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr int8_t kC2VDirMask = 1;
constexpr int8_t kV2CDirMask = 2;
constexpr int8_t kBidirectionalDirMask = 3;
constexpr int32_t kSingleDirectionSlotNum = 8;
constexpr int32_t kBidirectionalSlotNum = 4;

struct FrontendPipeHandles {
  Value c2vPipe;
  Value v2cPipe;
  Operation *anchorOp = nullptr;
};

using FrontendPipeHandleMap = llvm::DenseMap<int32_t, FrontendPipeHandles>;

template <typename InitOpT>
static LogicalResult requireFrontendGmSlotBuffer(InitOpT initOp) {
  if (initOp.getGmSlotBuffer())
    return success();
  return initOp.emitOpError("requires 'gm_slot_buffer' when lowering to a2/a3");
}

template <typename InitOpT>
static FailureOr<Value> createFrontendPipe(InitOpT initOp, IRRewriter &rewriter,
                                           PTOArch arch, Type pipeTy,
                                           int8_t dirMask, int32_t slotNum,
                                           Value localAddr,
                                           Value peerLocalAddr = Value{}) {
  Location loc = initOp.getLoc();
  auto dirAttr = rewriter.getI8IntegerAttr(dirMask);
  auto slotSizeAttr = rewriter.getI32IntegerAttr(initOp.getSlotSize());
  auto slotNumAttr = rewriter.getI32IntegerAttr(slotNum);
  auto noSplitAttr = initOp.getNosplitAttr();

  if (arch == PTOArch::A5) {
    auto pipe = rewriter.create<InitializeL2LPipeOp>(
        loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, IntegerAttr{},
        noSplitAttr, localAddr, peerLocalAddr);
    return pipe.getPipe();
  }

  if (failed(requireFrontendGmSlotBuffer(initOp)))
    return failure();

  IntegerAttr localSlotNumAttr = initOp.getLocalSlotNumAttr();
  if (!localSlotNumAttr)
    localSlotNumAttr = rewriter.getI32IntegerAttr(slotNum);
  auto pipe = rewriter.create<InitializeL2G2LPipeOp>(
      loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, localSlotNumAttr,
      IntegerAttr{}, noSplitAttr, initOp.getGmSlotBuffer(), localAddr,
      peerLocalAddr);
  return pipe.getPipe();
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles>
lowerSingleDirectionFrontendInit(InitOpT initOp, IRRewriter &rewriter,
                                 PTOArch arch, Type pipeTy, int8_t dirMask,
                                 Value localAddr) {
  auto pipeOr =
      createFrontendPipe(initOp, rewriter, arch, pipeTy, dirMask,
                         kSingleDirectionSlotNum, localAddr);
  if (failed(pipeOr))
    return failure();

  FrontendPipeHandles handles;
  if (dirMask == kC2VDirMask)
    handles.c2vPipe = *pipeOr;
  else
    handles.v2cPipe = *pipeOr;
  handles.anchorOp = pipeOr->getDefiningOp();
  return handles;
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles>
lowerBidirectionalFrontendInit(InitOpT initOp, IRRewriter &rewriter,
                               PTOArch arch, Type pipeTy) {
  auto pipeOr = createFrontendPipe(initOp, rewriter, arch, pipeTy,
                                   kBidirectionalDirMask,
                                   kBidirectionalSlotNum,
                                   initOp.getC2vConsumerBuf(),
                                   initOp.getV2cConsumerBuf());
  if (failed(pipeOr))
    return failure();

  FrontendPipeHandles handles;
  handles.c2vPipe = *pipeOr;
  handles.v2cPipe = *pipeOr;
  handles.anchorOp = pipeOr->getDefiningOp();
  return handles;
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles> lowerFrontendInitOp(InitOpT initOp,
                                                          IRRewriter &rewriter) {
  MLIRContext *ctx = initOp.getContext();
  auto pipeTy = PipeType::get(ctx);
  PTOArch arch = getTargetArch(initOp.getOperation());

  switch (initOp.getDirMask()) {
  case kC2VDirMask:
    return lowerSingleDirectionFrontendInit(initOp, rewriter, arch, pipeTy,
                                            kC2VDirMask,
                                            initOp.getC2vConsumerBuf());
  case kV2CDirMask:
    return lowerSingleDirectionFrontendInit(initOp, rewriter, arch, pipeTy,
                                            kV2CDirMask,
                                            initOp.getV2cConsumerBuf());
  case kBidirectionalDirMask:
    return lowerBidirectionalFrontendInit(initOp, rewriter, arch, pipeTy);
  default:
    return FrontendPipeHandles{};
  }
}

template <typename InitOpT>
static void propagateFrontendNoSplitAttr(InitOpT initOp,
                                         const FrontendPipeHandles &handles) {
  auto noSplitAttr = initOp.getNosplitAttr();
  if (!noSplitAttr)
    return;

  if (handles.anchorOp)
    handles.anchorOp->setAttr("nosplit", noSplitAttr);

  Operation *c2vOp =
      handles.c2vPipe ? handles.c2vPipe.getDefiningOp() : nullptr;
  Operation *v2cOp =
      handles.v2cPipe ? handles.v2cPipe.getDefiningOp() : nullptr;

  if (c2vOp && c2vOp != handles.anchorOp)
    c2vOp->setAttr("nosplit", noSplitAttr);
  if (v2cOp && v2cOp != handles.anchorOp && v2cOp != c2vOp)
    v2cOp->setAttr("nosplit", noSplitAttr);
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles> lowerAndEraseFrontendInit(InitOpT initOp,
                                                                IRRewriter &rewriter) {
  rewriter.setInsertionPoint(initOp);
  auto loweredOr = lowerFrontendInitOp(initOp, rewriter);
  if (failed(loweredOr))
    return failure();
  propagateFrontendNoSplitAttr(initOp, *loweredOr);
  rewriter.eraseOp(initOp);
  return *loweredOr;
}

static FailureOr<FrontendPipeHandleMap> lowerInitIfPresent(func::FuncOp funcOp,
                                                           IRRewriter &rewriter) {
  FrontendPipeHandleMap handlesById;
  SmallVector<Operation *> frontendInitOps;
  llvm::DenseMap<int32_t, Operation *> initOpById;
  bool hasDuplicateId = false;
  bool hasAicInit = false;
  bool hasAivInit = false;

  funcOp.walk([&](Operation *op) {
    if (auto init = dyn_cast<AicInitializePipeOp>(op)) {
      hasAicInit = true;
      frontendInitOps.push_back(op);
      auto [it, inserted] = initOpById.try_emplace(init.getId(), op);
      if (!inserted) {
        op->emitOpError()
            << "requires unique initialize_pipe id in function (duplicate id = "
            << init.getId() << ")";
        hasDuplicateId = true;
      }
      return WalkResult::advance();
    }
    if (auto init = dyn_cast<AivInitializePipeOp>(op)) {
      hasAivInit = true;
      frontendInitOps.push_back(op);
      auto [it, inserted] = initOpById.try_emplace(init.getId(), op);
      if (!inserted) {
        op->emitOpError()
            << "requires unique initialize_pipe id in function (duplicate id = "
            << init.getId() << ")";
        hasDuplicateId = true;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (hasDuplicateId)
    return failure();

  if (hasAicInit && hasAivInit) {
    funcOp.emitOpError("cannot mix pto.aic_initialize_pipe and "
                       "pto.aiv_initialize_pipe in one function");
    return failure();
  }

  for (Operation *op : frontendInitOps) {
    if (auto init = dyn_cast<AicInitializePipeOp>(op)) {
      int32_t id = init.getId();
      auto loweredOr = lowerAndEraseFrontendInit(init, rewriter);
      if (failed(loweredOr))
        return failure();
      handlesById.try_emplace(id, *loweredOr);
      continue;
    }

    auto init = cast<AivInitializePipeOp>(op);
    int32_t id = init.getId();
    auto loweredOr = lowerAndEraseFrontendInit(init, rewriter);
    if (failed(loweredOr))
      return failure();
    handlesById.try_emplace(id, *loweredOr);
  }

  return handlesById;
}

static bool hasFrontendPipeOps(func::FuncOp funcOp) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    if (isa<AicInitializePipeOp, AivInitializePipeOp, TPushToAivOp, TPushToAicOp,
            TPopFromAicOp, TPopFromAivOp, TFreeFromAicOp, TFreeFromAivOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static LogicalResult lowerFrontendDataOps(func::FuncOp funcOp,
                                          const FrontendPipeHandleMap &handlesById,
                                          IRRewriter &rewriter) {
  DominanceInfo dom(funcOp);
  SmallVector<Operation *> frontendOps;
  funcOp.walk([&](Operation *op) {
    if (isa<TPushToAivOp, TPushToAicOp, TPopFromAicOp, TPopFromAivOp,
            TFreeFromAicOp, TFreeFromAivOp>(op))
      frontendOps.push_back(op);
  });

  auto lookupHandles = [&](Operation *op, int32_t id)
      -> FailureOr<const FrontendPipeHandles *> {
    auto it = handlesById.find(id);
    if (it == handlesById.end()) {
      op->emitOpError()
          << "requires matching frontend initialize_pipe(id = " << id
          << ") in the same function";
      return failure();
    }
    const FrontendPipeHandles &handles = it->second;
    if (!handles.anchorOp || !dom.dominates(handles.anchorOp, op)) {
      op->emitOpError()
          << "requires dominating frontend initialize_pipe(id = " << id << ")";
      return failure();
    }
    return &handles;
  };

  for (Operation *op : frontendOps) {
    rewriter.setInsertionPoint(op);

    if (auto push = dyn_cast<TPushToAivOp>(op)) {
      auto handlesOr = lookupHandles(op, push.getId());
      if (failed(handlesOr))
        return failure();
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.c2vPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << push.getId()
                          << ") to enable C2V";
        return failure();
      }
      rewriter.replaceOpWithNewOp<TPushOp>(push, push.getTile(), handles.c2vPipe,
                                           push.getSplitAttr());
      continue;
    }

    if (auto push = dyn_cast<TPushToAicOp>(op)) {
      auto handlesOr = lookupHandles(op, push.getId());
      if (failed(handlesOr))
        return failure();
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.v2cPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << push.getId()
                          << ") to enable V2C";
        return failure();
      }
      rewriter.replaceOpWithNewOp<TPushOp>(push, push.getTile(), handles.v2cPipe,
                                           push.getSplitAttr());
      continue;
    }

    if (auto pop = dyn_cast<TPopFromAicOp>(op)) {
      auto handlesOr = lookupHandles(op, pop.getId());
      if (failed(handlesOr))
        return failure();
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.c2vPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << pop.getId()
                          << ") to enable C2V";
        return failure();
      }
      auto decl = rewriter.create<DeclareTileOp>(pop.getLoc(),
                                                 pop.getTile().getType());
      if (pop.getValidRow() && pop.getValidCol()) {
        rewriter.create<SetValidShapeOp>(pop.getLoc(), decl.getTile(),
                                         pop.getValidRow(), pop.getValidCol());
      }
      rewriter.create<TPopOp>(pop.getLoc(), decl.getTile(), handles.c2vPipe,
                              pop.getSplitAttr());
      rewriter.replaceOp(pop, decl.getTile());
      continue;
    }

    if (auto pop = dyn_cast<TPopFromAivOp>(op)) {
      auto handlesOr = lookupHandles(op, pop.getId());
      if (failed(handlesOr))
        return failure();
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.v2cPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << pop.getId()
                          << ") to enable V2C";
        return failure();
      }
      auto decl = rewriter.create<DeclareTileOp>(pop.getLoc(),
                                                 pop.getTile().getType());
      if (pop.getValidRow() && pop.getValidCol()) {
        rewriter.create<SetValidShapeOp>(pop.getLoc(), decl.getTile(),
                                         pop.getValidRow(), pop.getValidCol());
      }
      rewriter.create<TPopOp>(pop.getLoc(), decl.getTile(), handles.v2cPipe,
                              pop.getSplitAttr());
      rewriter.replaceOp(pop, decl.getTile());
      continue;
    }

    if (auto free = dyn_cast<TFreeFromAicOp>(op)) {
      auto handlesOr = lookupHandles(op, free.getId());
      if (failed(handlesOr))
        return failure();
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.c2vPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << free.getId()
                          << ") to enable C2V";
        return failure();
      }
      rewriter.replaceOpWithNewOp<TFreeOp>(free, handles.c2vPipe,
                                           free.getSplitAttr());
      continue;
    }

    auto free = cast<TFreeFromAivOp>(op);
    auto handlesOr = lookupHandles(op, free.getId());
    if (failed(handlesOr))
      return failure();
    const FrontendPipeHandles &handles = **handlesOr;
    if (!handles.v2cPipe) {
      op->emitOpError() << "requires initialize_pipe(id = " << free.getId()
                        << ") to enable V2C";
      return failure();
    }
    rewriter.replaceOpWithNewOp<TFreeOp>(free, handles.v2cPipe,
                                         free.getSplitAttr());
  }

  return success();
}

struct PTOLowerFrontendPipeOpsPass
    : public mlir::pto::impl::PTOLowerFrontendPipeOpsBase<
          PTOLowerFrontendPipeOpsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!hasFrontendPipeOps(funcOp))
      return;

    IRRewriter rewriter(funcOp.getContext());
    auto loweredOr = lowerInitIfPresent(funcOp, rewriter);
    if (failed(loweredOr)) {
      signalPassFailure();
      return;
    }

    if (failed(lowerFrontendDataOps(funcOp, *loweredOr, rewriter)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowerFrontendPipeOpsPass() {
  return std::make_unique<PTOLowerFrontendPipeOpsPass>();
}
