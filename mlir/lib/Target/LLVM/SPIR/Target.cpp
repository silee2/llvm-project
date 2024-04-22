//===- Target.cpp - MLIR LLVM SPIR target compilation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines SPIR target related functions including registration
// calls for the `#spir.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/SPIR/Target.h"

#include "mlir/Config/mlir-config.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/SPIRDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Target/LLVM/SPIR/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/SPIR/SPIRToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>

using namespace mlir;
using namespace mlir::spir;

namespace {
// Implementation of the `TargetAttrInterface` model.
class SPIRTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<SPIRTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

// Register the SPIR dialect, the SPIR translation and the target interface.
void mlir::spir::registerSPIRTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, spir::SPIRDialect *dialect) {
    SPIRTargetAttr::attachInterface<SPIRTargetAttrImpl>(*ctx);
  });
}

void mlir::spir::registerSPIRTargetInterfaceExternalModels(
    MLIRContext &context) {
  DialectRegistry registry;
  registerSPIRTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

SerializeGPUModuleBase::SerializeGPUModuleBase(
    Operation &module, SPIRTargetAttr target,
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
#if MLIR_ENABLE_SPIRV_CONVERSIONS
    const char *argv[] = {"llc", "--spirv-ext=+SPV_INTEL_vector_compute"};
    llvm::cl::ParseCommandLineOptions(2, argv, "SPIR-V backend");
    // If the `SPIRV` LLVM target was built, initialize it.
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTargetMC();
    LLVMInitializeSPIRVAsmPrinter();
#endif
  });
}

SPIRTargetAttr SerializeGPUModuleBase::getTarget() const { return target; }

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
  SPIRVSerializer(Operation &module, SPIRTargetAttr target,
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

SPIRVSerializer::SPIRVSerializer(Operation &module, SPIRTargetAttr target,
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

std::optional<std::string>
translateToSPIRVBinary(llvm::Module &llvmModule,
                       llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  llvm::raw_string_ostream stream(targetISA);

  { // Drop pstream after this to prevent the ISA from being stuck buffering
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;

    if (targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                          llvm::CodeGenFileType::ObjectFile))
      return std::nullopt;

    codegenPasses.run(llvmModule);
  }
  return stream.str();
}

std::optional<SmallVector<char, 0>>
SPIRVSerializer::moduleToObject(llvm::Module &llvmModule) {
  // Return LLVM IR if the compilation target is offload.
#define DEBUG_TYPE "serialize-spir-to-llvm"
  LLVM_DEBUG({ llvm::dbgs() << llvmModule << "\n"; });
#undef DEBUG_TYPE
  bool needVectorCompute = false;
  if (getTarget().getVce()) {
    for (const spirv::Extension &exts : getTarget().getVce()->getExtensions()) {
      if (exts == spirv::Extension::SPV_INTEL_vector_compute) {
        needVectorCompute = true;
      }
    }
  }
  if (needVectorCompute) {
    auto &fList = llvmModule.getFunctionList();
    for (auto &fRef : fList) {
      // For each SPIR_KERNEL and non-builtin SPIR_FUNC
      // Append named metadata at end of module
      // Set    (A) !spirv.ExecutionMode with subgroupsize (35) = 1
      //     or (B) function decorator !intel_reqd_sub_group_size !5
      // Both works for SPIR-V backend and translator
      // Set    (C) !spirv.Decorations VectorComputeFunctionINTEL (5626)
      //     or (D) function attribute "VCFunction"
      // SPIR-V backend does not have support and needs to be added
      // function attribute will be easier to add
      // Using (A) and (D) for now
      // SubgroupSize is only set for SPIR_KERNEL
      if (fRef.getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
        llvm::NamedMDNode *node =
            llvmModule.getOrInsertNamedMetadata("spirv.ExecutionMode");
        llvm::Metadata *llvmMetadata[] = {
            llvm::ValueAsMetadata::get(&fRef),
            llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(llvmModule.getContext()), 35)),
            llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(llvmModule.getContext()), 1))};
        llvm::MDNode *llvmMetadataNode =
            llvm::MDNode::get(llvmModule.getContext(), llvmMetadata);
        node->addOperand(llvmMetadataNode);
        fRef.addFnAttr("VCFunction");
      } else if ((fRef.getCallingConv() == llvm::CallingConv::SPIR_FUNC) &&
                 fRef.getName().starts_with("llvm.genx.")) {
        fRef.addFnAttr("VCFunction");
      }
    }
  }
  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Offload)
    return SerializeGPUModuleBase::moduleToObject(llvmModule);

  std::optional<llvm::TargetMachine *> targetMachine =
      getOrCreateTargetMachine();
  if (!targetMachine) {
    getOperation().emitError() << "Target Machine unavailable for triple "
                               << triple << ", can't compile with LLVM\n";
    return std::nullopt;
  }

#if MLIR_ENABLE_SPIRV_CONVERSIONS
  if (targetOptions.getCompilationTarget() ==
      gpu::CompilationTarget::Assembly) {
    // Translate the Module to ISA which is SPIR-V text format.
    std::optional<std::string> serializedISA =
        translateToISA(llvmModule, **targetMachine);
    if (!serializedISA) {
      getOperation().emitError() << "Failed translating the module to ISA.";
      return std::nullopt;
    }
#define DEBUG_TYPE "serialize-spir-to-isa"
    LLVM_DEBUG({ llvm::dbgs() << *serializedISA << "\n"; });
#undef DEBUG_TYPE
    // Return ISA assembly code if the compilation _target is assembly.
    return SmallVector<char, 0>(serializedISA->begin(), serializedISA->end());
  }

  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Binary) {
    // Translate the Module to ISA binary which is SPIR-V binary format.
    std::optional<std::string> serializedSPIRVBinary =
        translateToSPIRVBinary(llvmModule, **targetMachine);
    if (!serializedSPIRVBinary) {
      getOperation().emitError() << "Failed translating the module to Binary.";
      return std::nullopt;
    }
#define DEBUG_TYPE "serialize-spir-to-binary"
    LLVM_DEBUG({ llvm::dbgs() << *serializedSPIRVBinary << "\n"; });
#undef DEBUG_TYPE
    // Return ISA assembly code if the compilation target is assembly.
    return SmallVector<char, 0>(serializedSPIRVBinary->begin(),
                                serializedSPIRVBinary->end());
  }
#endif // MLIR_SPIRV_CONVERSIONS_ENABLED

  // Compilation target 'fatbin' is not supported.
  getOperation().emitError() << "Compilation target is not supported. "
                             << "Use different compilation target.";
  return std::nullopt;
}

std::optional<SmallVector<char, 0>>
SPIRTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                      const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("Module must be a GPU module.");
    return std::nullopt;
  }
  SPIRVSerializer serializer(*module, cast<SPIRTargetAttr>(attribute), options);
  serializer.init();
  return serializer.run();
}

Attribute
SPIRTargetAttrImpl::createObject(Attribute attribute,
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
