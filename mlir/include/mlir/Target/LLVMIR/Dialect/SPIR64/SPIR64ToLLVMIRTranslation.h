//===- SPIR64ToLLVMIRTranslation.h - SPIR64 to LLVM IR ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for SPIR64 dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_SPIR64_SPIR64TOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_SPIR64_SPIR64TOLLVMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the SPIR64 dialect and the translation from it to the LLVM IR in
/// the given registry;
void registerSPIR64DialectTranslation(DialectRegistry &registry);

/// Register the SPIR64 dialect and the translation from it in the registry
/// associated with the given context.
void registerSPIR64DialectTranslation(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_SPIR64_SPIRVTOLLVMIRTRANSLATION_H
