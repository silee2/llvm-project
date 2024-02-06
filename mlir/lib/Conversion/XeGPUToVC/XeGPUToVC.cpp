//===- XeGPUToVC.cpp - XeGPU to vc intrinsic conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/XeGPUToVC/XeGPUToVC.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTXEGPUTOVC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::xegpu;

struct ConvertXeGPUToVCPass
    : public impl::ConvertXeGPUToVCBase<ConvertXeGPUToVCPass> {
  ConvertXeGPUToVCPass() = default;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    LLVMTypeConverter converter(ctx);
    populateXeGPUToVCConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<::mlir::xegpu::XeGPUDialect>();
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

void mlir::populateXeGPUToVCConversionPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns) {
}

std::unique_ptr<Pass> mlir::createConvertXeGPUToVCPass() {
  return std::make_unique<ConvertXeGPUToVCPass>();
}
