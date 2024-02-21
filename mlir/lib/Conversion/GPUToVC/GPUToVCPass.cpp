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
/*
    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    builder.setInsertionPoint(m.getBody(),
                              m.getBody()->begin());
    // instead of default builder, need a wrapper that checks GPU and name
    //builder.create<LLVM::CallIntrinsicOp>(m.getLoc(),
    llvm::GenXIntrinsic::ID id = llvm::GenXIntrinsic::lookupGenXIntrinsicID("llvm.genx.simdcf.get.em");
    std::cout << "GenXIntrinsic ID: " << id << std::endl;
    bool isSup = llvm::GenXIntrinsic::isSupportedPlatform("XeLP", id);
    std::cout << "Is supported platform: " << (isSup ? "y" : "n") << std::endl;
    if(!llvm::GenXIntrinsic::isGenXIntrinsic(id))
        signalPassFailure();
*/

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
      populateGpuRewritePatterns(patterns);
      // Convert gpu index ops to func.call to OCL builtins
      // populateIndexOCLRewritePatterns(patterns);
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

