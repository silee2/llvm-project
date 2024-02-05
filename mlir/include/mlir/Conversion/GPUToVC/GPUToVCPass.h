//===- GPUToVCPass.h - Convert GPU kernel to vc intrinsics ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOVC_GPUTOVCPASS_H_
#define MLIR_CONVERSION_GPUTOVC_GPUTOVCPASS_H_

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class ConversionTarget;
class RewritePatternSet;
class Pass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

#define GEN_PASS_DECL_CONVERTGPUTOVC
#include "mlir/Conversion/Passes.h.inc"

/// Configure target to convert from the GPU dialect to VC.
void configureGpuToVCConversionLegality(ConversionTarget &target);

/// Collect a set of patterns to convert from the GPU dialect to VC.
void populateGpuToVCConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOVC_GPUTOVCPASS_H_
