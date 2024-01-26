//===- Utils.h - MLIR SPIR64 target utils -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files declares SPIR64 target related utility classes and functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_SPIR64_UTILS_H
#define MLIR_TARGET_LLVM_SPIR64_UTILS_H

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/SPIR64Dialect.h"
#include "mlir/Target/LLVM/ModuleToObject.h"

namespace mlir {
namespace spir64 {

/// Base class for all SPIR64 serializations from GPU modules into binary strings.
/// By default this class serializes into LLVM bitcode.
class SerializeGPUModuleBase : public LLVM::ModuleToObject {
public:
  /// Initializes the `toolkitPath` with the path in `targetOptions` or if empty
  /// with the path in `getCUDAToolkitPath`.
  SerializeGPUModuleBase(Operation &module, SPIR64TargetAttr target,
                         const gpu::TargetOptions &targetOptions = {});

  /// Initializes the LLVM NVPTX target by safely calling `LLVMInitializeNVPTX*`
  /// methods if available.
  static void init();

  /// Returns the target attribute.
  SPIR64TargetAttr getTarget() const;

  /// Returns the bitcode files to be loaded.
  ArrayRef<std::string> getFileList() const;

  /// Loads the bitcode files in `fileList`.
  virtual std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
  loadBitcodeFiles(llvm::Module &module) override;

protected:
  /// SPIR64 target attribute.
  SPIR64TargetAttr target;

  /// List of LLVM bitcode files to link to.
  SmallVector<std::string> fileList;
};
} // namespace spir64
} // namespace mlir

#endif // MLIR_TARGET_LLVM_SPIR64_UTILS_H
