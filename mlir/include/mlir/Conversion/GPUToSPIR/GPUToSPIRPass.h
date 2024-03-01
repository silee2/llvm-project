//===- GPUToSPIRPass.h - Convert GPU kernel to LLVM SPIR ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOSPIR_GPUTOSPIRPASS_H_
#define MLIR_CONVERSION_GPUTOSPIR_GPUTOSPIRPASS_H_

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

#define GEN_PASS_DECL_CONVERTGPUTOSPIR
#include "mlir/Conversion/Passes.h.inc"

/// Configure target to convert from the GPU dialect to SPIR.
void configureGpuToSPIRConversionLegality(ConversionTarget &target);

/// Collect a set of patterns to convert from the GPU dialect to SPIR.
void populateGpuToSPIRConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOSPIR_GPUTOSPIRPASS_H_
