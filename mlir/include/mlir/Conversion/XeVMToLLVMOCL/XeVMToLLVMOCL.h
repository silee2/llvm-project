//===-- XeVMToLLVMOCL.h - Convert XeVM to OpenCL extensions -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_XEVMTOLLVMOCL_XEVMTOLLVMOCLPASS_H_
#define MLIR_CONVERSION_XEVMTOLLVMOCL_XEVMTOLLVMOCLPASS_H_

#include <memory>

namespace mlir {
class ConversionTarget;
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTXEVMTOLLVMOCLPASS
#include "mlir/Conversion/Passes.h.inc"

void populateXeVMToLLVMOCLConversionPatterns(ConversionTarget &target,
                                             RewritePatternSet &patterns);

void registerConvertXeVMToLLVMOCLInterface(DialectRegistry &registry);
} // namespace mlir

#endif // MLIR_CONVERSION_XEVMTOLLVMOCL_XEVMTOLLVMOCLPASS_H_
