//===- SPIRToLLVMIRTranslation.cpp - Translate SPIR to LLVM IR ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR SPIR dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/SPIR/SPIRToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/SPIRDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

using namespace mlir;
using namespace mlir::LLVM;

void mlir::registerSPIRDialectTranslation(DialectRegistry &registry) {
  registry.insert<spir::SPIRDialect>();
}

void mlir::registerSPIRDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerSPIRDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
