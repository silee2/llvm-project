//===- SPIRAttachTarget.cpp - Attach an SPIR target
//-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GpuSPIRAttachTarget` pass, attaching
// `#nvvm.target` attributes to GPU modules.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/SPIRDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVM/SPIR/Target.h"
#include "llvm/Support/Regex.h"

namespace mlir {
#define GEN_PASS_DEF_GPUSPIRATTACHTARGET
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::spir;
using namespace mlir::spirv;

namespace {
struct SPIRAttachTarget
    : public impl::GpuSPIRAttachTargetBase<SPIRAttachTarget> {
  using Base::Base;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spir::SPIRDialect>();
    registry.insert<spirv::SPIRVDialect>();
  }
};
} // namespace

void SPIRAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());
  ArrayRef<std::string> libs(linkLibs);
  SmallVector<StringRef> filesToLink(libs.begin(), libs.end());

  UnitAttr unitAttr = builder.getUnitAttr();
  SmallVector<NamedAttribute, 6> flags;
  auto addFlag = [&](StringRef flag) {
    flags.push_back(builder.getNamedAttr(flag, unitAttr));
  };
  for (const auto &flag : targetFlags) {
    addFlag(flag);
  }

  auto versionSymbol = symbolizeVersion(spirvVersion);
  if (!versionSymbol)
    return signalPassFailure();
  Version version = versionSymbol.value();
  SmallVector<Capability, 4> capabilities;
  SmallVector<Extension, 8> extensions;
  for (const auto &cap : spirvCapabilities) {
    auto capSymbol = symbolizeCapability(cap);
    if (capSymbol)
      capabilities.push_back(capSymbol.value());
  }
  ArrayRef<Capability> caps(capabilities);
  for (const auto &ext : spirvExtensions) {
    auto extSymbol = symbolizeExtension(ext);
    if (extSymbol)
      extensions.push_back(extSymbol.value());
  }
  ArrayRef<Extension> exts(extensions);
  VerCapExtAttr vce = VerCapExtAttr::get(version, caps, exts, &getContext());

  auto target = builder.getAttr<SPIRTargetAttr>(
      optLevel, triple, chip, features,
      flags.empty() ? nullptr : builder.getDictionaryAttr(flags),
      filesToLink.empty() ? nullptr : builder.getStrArrayAttr(filesToLink),
      vce);
  llvm::Regex matcher(moduleMatcher);
  for (Region &region : getOperation()->getRegions())
    for (Block &block : region.getBlocks())
      for (auto module : block.getOps<gpu::GPUModuleOp>()) {
        // Check if the name of the module matches.
        if (!moduleMatcher.empty() && !matcher.match(module.getName()))
          continue;
        // Create the target array.
        SmallVector<Attribute> targets;
        if (std::optional<ArrayAttr> attrs = module.getTargets())
          targets.append(attrs->getValue().begin(), attrs->getValue().end());
        targets.push_back(target);
        // Remove any duplicate targets.
        targets.erase(std::unique(targets.begin(), targets.end()),
                      targets.end());
        // Update the target attribute array.
        module.setTargetsAttr(builder.getArrayAttr(targets));
      }
}
