#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOWRAPFUNCTIONSINSECTIONS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool hasExistingSection(func::FuncOp funcOp) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    if (isa<SectionCubeOp, SectionVectorOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

template <typename SectionOpT>
static void wrapSingleBlockFuncBody(func::FuncOp funcOp) {
  Block &entryBlock = funcOp.getBody().front();
  Operation *terminator = entryBlock.getTerminator();

  OpBuilder builder(terminator);
  auto sectionOp = builder.create<SectionOpT>(funcOp.getLoc());
  sectionOp.getBody().push_back(new Block());
  Block &sectionBlock = sectionOp.getBody().front();

  auto sectionIt = Block::iterator(sectionOp.getOperation());
  sectionBlock.getOperations().splice(sectionBlock.end(),
                                      entryBlock.getOperations(),
                                      entryBlock.begin(), sectionIt);
}

static LogicalResult rewriteFunction(func::FuncOp funcOp) {
  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(
          FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return success();

  if (!funcOp.getBody().hasOneBlock())
    return funcOp.emitOpError(
        "requires a single-block body for kernel_kind wrapping");

  if (hasExistingSection(funcOp)) {
    return funcOp.emitOpError(
        "already contains pto.section.cube or pto.section.vector");
  }

  switch (kernelKindAttr.getKernelKind()) {
  case FunctionKernelKind::Cube:
    wrapSingleBlockFuncBody<SectionCubeOp>(funcOp);
    return success();
  case FunctionKernelKind::Vector:
    wrapSingleBlockFuncBody<SectionVectorOp>(funcOp);
    return success();
  }

  llvm_unreachable("unexpected kernel kind");
}

struct PTOWrapFunctionsInSectionsPass
    : public mlir::pto::impl::PTOWrapFunctionsInSectionsBase<
          PTOWrapFunctionsInSectionsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (failed(rewriteFunction(funcOp)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOWrapFunctionsInSectionsPass() {
  return std::make_unique<PTOWrapFunctionsInSectionsPass>();
}
