//===- XeGPUToVC.h - Convert XeGPU to vc intrinsic --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_XEGPUTOVC_XEGPUTOVC_H_
#define MLIR_CONVERSION_XEGPUTOVC_XEGPUTOVC_H_

#include <memory>
#include <string>

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTXEGPUTOVC
#include "mlir/Conversion/Passes.h.inc"

void populateXeGPUToVCConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertXeGPUToVCPass();

} // namespace mlir

#endif // MLIR_CONVERSION_XEGPUTOVC_XEGPUTOVC_H_
