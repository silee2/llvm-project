//===- SPIR64ToLLVMIRTranslation.cpp - Translate SPIR64 to LLVM IR ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR SPIR64 dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/SPIR64/SPIR64ToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/SPIR64Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

using namespace mlir;
using namespace mlir::LLVM;

void mlir::registerSPIR64DialectTranslation(DialectRegistry &registry) {
  registry.insert<spir64::SPIR64Dialect>();
}

void mlir::registerSPIR64DialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerSPIR64DialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
