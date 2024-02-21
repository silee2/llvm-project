//===- GPUToVCPass.cpp - MLIR GPU to VC lowering passes -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate VC operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"

#include "mlir/Conversion/GPUToVC/GPUToVCPass.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"
#include <optional>

#include "llvm/GenXIntrinsics/GenXIntrinsics.h"

#include <iostream>

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUTOVC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static FlatSymbolRefAttr getFuncRefAttr(gpu::GPUModuleOp module, StringRef name,
                                               TypeRange resultType,
                                               ValueRange operands,
                                               bool emitCInterface) {
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<func::FuncOp>(
        module.getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));
        //FunctionType::get(context, operands.getTypes(), {moduleBuilder.getIndexType()}));
    func.setPrivate();
    if (emitCInterface)
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));
  }
  return result;
}

static func::CallOp createFuncCall(
    PatternRewriter &rewriter, Location loc, StringRef name, TypeRange resultType,
    ValueRange operands, bool emitCInterface) {
  auto module = rewriter.getBlock()->getParentOp()->getParentOfType<gpu::GPUModuleOp>();
  FlatSymbolRefAttr fn =
      getFuncRefAttr(module, name, resultType, operands, emitCInterface);
  return rewriter.create<func::CallOp>(loc, fn, resultType, operands);
}

template <typename Op, const char * FName>
struct GPUIndexIntrinsicOpToOCLBuiltinLowering : public OpRewritePattern<Op> {
    public:
        using OpRewritePattern<Op>::OpRewritePattern;
    LogicalResult
    matchAndRewrite(Op op,
            PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();

    Value argOp;
    Type i32Ty = IntegerType::get(context, 32);
    Type indexTy = rewriter.getIndexType();
    switch (op.getDimension()) {
    case gpu::Dimension::x:
      argOp = rewriter.createOrFold<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i32Ty, 0));
      break;
    case gpu::Dimension::y:
      argOp = rewriter.createOrFold<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i32Ty, 1));
      break;
    case gpu::Dimension::z:
      argOp = rewriter.createOrFold<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i32Ty, 2));
      break;
    }
    llvm::SmallVector<mlir::Value> operands{argOp};
    //TypeRange resultTypes{indexTy};
    auto newOp = createFuncCall(rewriter, loc, FName, TypeRange{indexTy}, operands, false);
    rewriter.replaceOp(op, newOp);
    return success();
    }
};


static const char get_global_id[] = "_Z13get_global_idj";
static const char get_local_id[] = "_Z12get_local_idj";
static const char get_local_size[] = "_Z12get_local_sizej";
static const char get_group_id[] = "_Z12get_group_idj";
static const char get_num_groups[] = "_Z14get_num_groupsj";

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding VC intrinsics equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct GPUToVCPass
    : public impl::ConvertGPUToVCBase<GPUToVCPass> {
  using Base::Base;

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    // Request C wrapper emission.
    for (auto func : m.getOps<func::FuncOp>()) {
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(&getContext()));
    }

    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));

    // Apply in-dialect lowering. In-dialect lowering will replace
    // ops which need to be lowered further, which is not supported by a
    // single conversion pass.
    {
      RewritePatternSet patterns(m.getContext());
      // Convert gpu index ops to func.call to OCL builtins
      patterns.add<
          GPUIndexIntrinsicOpToOCLBuiltinLowering<gpu::ThreadIdOp, get_local_id>,
          GPUIndexIntrinsicOpToOCLBuiltinLowering<gpu::BlockDimOp, get_local_size>,
          GPUIndexIntrinsicOpToOCLBuiltinLowering<gpu::BlockIdOp, get_group_id>,
          GPUIndexIntrinsicOpToOCLBuiltinLowering<gpu::GridDimOp, get_num_groups>,
          GPUIndexIntrinsicOpToOCLBuiltinLowering<gpu::GlobalIdOp, get_global_id>
          >(patterns.getContext());
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
        return signalPassFailure();
    }

    LLVMTypeConverter converter(m.getContext(), options);
    RewritePatternSet llvmPatterns(m.getContext());

    arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    llvmPatterns.add<GPUFuncOpLowering>(converter, 0, 0, builder.getStringAttr("spir_func"));
    llvmPatterns.add<GPUReturnOpLowering>(converter);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
  }
};

} // namespace

