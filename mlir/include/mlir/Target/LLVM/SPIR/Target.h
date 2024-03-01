//===- Target.h - MLIR SPIR target registration -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the SPIR target interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_SPIR_TARGET_H
#define MLIR_TARGET_LLVM_SPIR_TARGET_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
namespace spir {
/// Registers the `TargetAttrInterface` for the `#nvvm.target` attribute in the
/// given registry.
void registerSPIRTargetInterfaceExternalModels(DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#nvvm.target` attribute in the
/// registry associated with the given context.
void registerSPIRTargetInterfaceExternalModels(MLIRContext &context);
} // namespace spir
} // namespace mlir

#endif // MLIR_TARGET_LLVM_SPIR_TARGET_H
