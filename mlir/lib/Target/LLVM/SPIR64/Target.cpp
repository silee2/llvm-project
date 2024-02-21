//===- Target.cpp - MLIR LLVM SPIR64 target compilation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines SPIR64 target related functions including registration
// calls for the `#spir.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/SPIR64/Target.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/SPIR64Dialect.h"
#include "mlir/Target/LLVM/SPIR64/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/SPIR64/SPIR64ToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Support/TargetSelect.h"

#if MLIR_SPIRV_CONVERSIONS_ENABLED == 1 &&                                     \
    MLIR_SPIRV_LLVM_TRANSLATOR_ENABLED == 1
#include "LLVMSPIRVLib.h"
#include "spirv-tools/libspirv.hpp"
#endif

#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>

using namespace mlir;
using namespace mlir::spir64;

namespace {
// Implementation of the `TargetAttrInterface` model.
class SPIR64TargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<SPIR64TargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

// Register the SPIR64 dialect, the SPIR64 translation and the target interface.
void mlir::spir64::registerSPIR64TargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, spir64::SPIR64Dialect *dialect) {
    SPIR64TargetAttr::attachInterface<SPIR64TargetAttrImpl>(*ctx);
  });
}

void mlir::spir64::registerSPIR64TargetInterfaceExternalModels(
    MLIRContext &context) {
  DialectRegistry registry;
  registerSPIR64TargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

SerializeGPUModuleBase::SerializeGPUModuleBase(
    Operation &module, SPIR64TargetAttr target,
    const gpu::TargetOptions &targetOptions)
    : ModuleToObject(module, target.getTriple(), "", target.getFeatures(),
                     target.getO()),
      target(target), fileList(targetOptions.getLinkFiles()) {

  // Append the files in the target attribute.
  if (ArrayAttr files = target.getLink())
    for (Attribute attr : files.getValue())
      if (auto file = dyn_cast<StringAttr>(attr))
        fileList.push_back(file.str());
}

void SerializeGPUModuleBase::init() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
#if MLIR_SPIRV_CONVERSIONS_ENABLED == 1
    // If the `SPIRV` LLVM target was built, initialize it.
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTargetMC();
    LLVMInitializeSPIRVAsmPrinter();
#endif
  });
}

SPIR64TargetAttr SerializeGPUModuleBase::getTarget() const { return target; }

ArrayRef<std::string> SerializeGPUModuleBase::getFileList() const {
  return fileList;
}

std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
SerializeGPUModuleBase::loadBitcodeFiles(llvm::Module &module) {
  SmallVector<std::unique_ptr<llvm::Module>> bcFiles;
  if (failed(loadBitcodeFilesFromList(module.getContext(), fileList, bcFiles,
                                      true)))
    return std::nullopt;
  return std::move(bcFiles);
}

namespace {
class SPIRVSerializer : public SerializeGPUModuleBase {
public:
  SPIRVSerializer(Operation &module, SPIR64TargetAttr target,
                   const gpu::TargetOptions &targetOptions);

  gpu::GPUModuleOp getOperation();

  std::optional<SmallVector<char, 0>>
  compileToBinary(const std::string &serializedISA);

  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override;

private:
  // Target options.
  gpu::TargetOptions targetOptions;
};
} // namespace

SPIRVSerializer::SPIRVSerializer(Operation &module, SPIR64TargetAttr target,
                                   const gpu::TargetOptions &targetOptions)
    : SerializeGPUModuleBase(module, target, targetOptions),
      targetOptions(targetOptions) {}

gpu::GPUModuleOp SPIRVSerializer::getOperation() {
  return dyn_cast<gpu::GPUModuleOp>(&SerializeGPUModuleBase::getOperation());
}

std::optional<SmallVector<char, 0>>
compileToBinary(const std::string &serializedISA) {
  return std::nullopt;
}

std::optional<SmallVector<char, 0>>
SPIRVSerializer::moduleToObject(llvm::Module &llvmModule) {
  // Return LLVM IR if the compilation target is offload.
#define DEBUG_TYPE "serialize-to-llvm"
  LLVM_DEBUG({
    llvm::dbgs() << "LLVM IR for module: " << getOperation().getNameAttr()
                 << "\n"
                 << llvmModule << "\n";
  });
#undef DEBUG_TYPE
  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Offload)
    return SerializeGPUModuleBase::moduleToObject(llvmModule);

#if MLIR_SPIRV_CONVERSIONS_ENABLED == 1 && MLIR_SPIRV_LLVM_TRANSLATOR_ENABLED == 0
  std::optional<llvm::TargetMachine *> targetMachine =
      getOrCreateTargetMachine();
  if (!targetMachine) {
    getOperation().emitError() << "Target Machine unavailable for triple "
                               << triple << ", can't compile with LLVM\n";
    return std::nullopt;
  }
#endif

#if MLIR_SPIRV_CONVERSIONS_ENABLED == 1
  if (targetOptions.getCompilationTarget() ==
      gpu::CompilationTarget::Assembly) {
#if MLIR_SPIRV_LLVM_TRANSLATOR_ENABLED == 0
    // Translate the Module to ISA which is SPIR-V text format.
    std::optional<std::string> serializedISA =
        translateToISA(llvmModule, **targetMachine);
    if (!serializedISA) {
      getOperation().emitError() << "Failed translating the module to ISA.";
      return std::nullopt;
    }
#define DEBUG_TYPE "serialize-to-isa"
    LLVM_DEBUG({
      llvm::dbgs() << "ISA for module: " << getOperation().getNameAttr() << "\n"
                   << *serializedISA << "\n";
    });
#undef DEBUG_TYPE
    // Return ISA assembly code if the compilation target is assembly.
    return SmallVector<char, 0>(serializedISA->begin(), serializedISA->end());
#else
    spvtools::SpirvTools spvTool(SPV_ENV_OPENCL_2_0);
    std::string err;
    std::ostringstream outStream;
    bool Success = writeSpirv(&llvmModule, outStream, err);
    if (!Success) {
      getOperation().emitError()
          << "Failed translating the module to ISA. " << err;
      return std::nullopt;
    }
    std::string serializedISA;
    if (!spvTool.Disassemble(
            reinterpret_cast<const uint32_t *>(outStream.str().data()),
            outStream.str().size() / sizeof(uint32_t), &serializedISA)) {
      getOperation().emitError() << "Failed translating the module to ISA.";
      return std::nullopt;
    }
#define DEBUG_TYPE "serialize-to-isa"
    LLVM_DEBUG({
      llvm::dbgs() << "ISA for module: " << getOperation().getNameAttr() << "\n"
                   << serializedISA << "\n";
    });
#undef DEBUG_TYPE
    return SmallVector<char, 0>(serializedISA.begin(), serializedISA.end());
#endif // MLIR_SPIRV_LLVM_TRANSLATOR_ENABLED == 0
  }

  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Binary) {
#if MLIR_SPIRV_LLVM_TRANSLATOR_ENABLED == 0
    // Translate the Module to ISA binary which is SPIR-V binary format.
    std::optional<std::string> serializedISABinary =
        translateToISABinary(llvmModule, **targetMachine);
    if (!serializedISABinary) {
      getOperation().emitError() << "Failed translating the module to Binary.";
      return std::nullopt;
    }
#define DEBUG_TYPE "serialize-to-binary"
    LLVM_DEBUG({
      llvm::dbgs() << "ISA binary for module: " << getOperation().getNameAttr()
                   << "\n"
                   << *serializedISABinary << "\n";
    });
#undef DEBUG_TYPE
    // Return ISA assembly code if the compilation target is assembly.
    return SmallVector<char, 0>(serializedISABinary->begin(),
                                serializedISABinary->end());
#else
    std::string err;
    std::ostringstream outStream;
    bool Success = writeSpirv(&llvmModule, outStream, err);
    if (!Success) {
      getOperation().emitError()
          << "Failed translating the module to Binary. " << err;
      return std::nullopt;
    }
    std::string serializedISABinary = outStream.str();
#define DEBUG_TYPE "serialize-to-binary"
    LLVM_DEBUG({
      llvm::dbgs() << "ISA binary for module: " << getOperation().getNameAttr()
                   << "\n"
                   << serializedISABinary << "\n";
    });
#undef DEBUG_TYPE
    return SmallVector<char, 0>(serializedISABinary.begin(), serializedISABinary.end());
#endif // MLIR_SPIRV_LLVM_TRANSLATOR_ENABLED == 0
  }
#endif // MLIR_SPIRV_CONVERSIONS_ENABLED

  // Compilation target 'fatbin' is not supported.
  getOperation().emitError() << "Compilation target is not supported. "
                             << "Use different compilation target.";
  return std::nullopt;
}

std::optional<SmallVector<char, 0>> SPIR64TargetAttrImpl::serializeToObject(
    Attribute attribute, Operation *module,
    const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("Module must be a GPU module.");
    return std::nullopt;
  }
  SPIRVSerializer serializer(*module, cast<SPIR64TargetAttr>(attribute),
                              options);
  serializer.init();
  return serializer.run();
}

Attribute
SPIR64TargetAttrImpl::createObject(Attribute attribute,
                                  const SmallVector<char, 0> &object,
                                  const gpu::TargetOptions &options) const {
  gpu::CompilationTarget format = options.getCompilationTarget();
  Builder builder(attribute.getContext());
  return builder.getAttr<gpu::ObjectAttr>(
      attribute,
      format > gpu::CompilationTarget::Binary ? gpu::CompilationTarget::Binary
                                              : format,
      builder.getStringAttr(StringRef(object.data(), object.size())), nullptr);
}
