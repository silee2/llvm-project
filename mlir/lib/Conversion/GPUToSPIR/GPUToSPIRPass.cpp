//===- GPUToSPIRPass.cpp - MLIR GPU to VC lowering passes
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate VC operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Conversion/GPUToSPIR/GPUToSPIRPass.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"
#include <optional>

#include <iostream>

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUTOSPIR
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static FlatSymbolRefAttr getFuncRefAttr(gpu::GPUModuleOp module, StringRef name,
                                        TypeRange resultType,
                                        ValueRange operands,
                                        bool emitCInterface) {
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<func::FuncOp>(
        module.getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));
    // FunctionType::get(context, operands.getTypes(),
    // {moduleBuilder.getIndexType()}));
    func.setPrivate();
    if (emitCInterface)
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));
  }
  return result;
}

static func::CallOp createFuncCall(PatternRewriter &rewriter, Location loc,
                                   StringRef name, TypeRange resultType,
                                   ValueRange operands, bool emitCInterface) {
  auto module =
      rewriter.getBlock()->getParentOp()->getParentOfType<gpu::GPUModuleOp>();
  FlatSymbolRefAttr fn =
      getFuncRefAttr(module, name, resultType, operands, emitCInterface);
  return rewriter.create<func::CallOp>(loc, fn, resultType, operands);
}

template <typename Op, const char *FName>
struct GPUIndexIntrinsicOpToOCLBuiltinLowering : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();

    Value argOp;
    Type i32Ty = IntegerType::get(context, 32);
    Type indexTy = rewriter.getIndexType();
    switch (op.getDimension()) {
    case gpu::Dimension::x:
      argOp = rewriter.createOrFold<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(i32Ty, 0));
      break;
    case gpu::Dimension::y:
      argOp = rewriter.createOrFold<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(i32Ty, 1));
      break;
    case gpu::Dimension::z:
      argOp = rewriter.createOrFold<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(i32Ty, 2));
      break;
    }
    llvm::SmallVector<mlir::Value> operands{argOp};
    // TypeRange resultTypes{indexTy};
    auto newOp = createFuncCall(rewriter, loc, FName, TypeRange{indexTy},
                                operands, false);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

template <typename Op, const char *FName>
struct GPUSubgroupIndexIntrinsicOpToOCLBuiltinLowering
    : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();

    Type i32Ty = IntegerType::get(context, 32);
    llvm::SmallVector<mlir::Value> operands;
    func::CallOp newOp =
        createFuncCall(rewriter, loc, FName, TypeRange{i32Ty}, operands, false);
    Type indexTy = rewriter.getIndexType();
    if (i32Ty != indexTy) {
      Value castVal = rewriter.create<arith::IndexCastOp>(op->getLoc(), indexTy,
                                                          newOp->getResult(0));
      rewriter.replaceOp(op, castVal);
    } else {
      rewriter.replaceOp(op, newOp);
    }
    return success();
  }
};

static IntegerAttr wrapNumericMemorySpace(MLIRContext *ctx, unsigned space) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), space);
}

void populateSPIRVMemorySpaceAttributeConversions(
    TypeConverter &typeConverter,
    const std::function<unsigned(spirv::StorageClass)> &mapping) {
  typeConverter.addTypeAttributeConversion(
      [mapping](BaseMemRefType type, spirv::StorageClassAttr memorySpaceAttr) {
        spirv::StorageClass memorySpace = memorySpaceAttr.getValue();
        unsigned addressSpace = mapping(memorySpace);
        return wrapNumericMemorySpace(memorySpaceAttr.getContext(),
                                      addressSpace);
      });
}

// static const char get_global_id[] = "_Z13get_global_idj";
static const char get_global_id[] = "_Z33__spirv_BuiltInGlobalInvocationIdi";
// static const char get_local_id[] = "_Z12get_local_idj";
static const char get_local_id[] = "_Z32__spirv_BuiltInLocalInvocationIdi";
// static const char get_local_size[] = "_Z12get_local_sizej";
static const char get_local_size[] = "_Z28__spirv_BuiltInWorkgroupSizei";
// static const char get_group_id[] = "_Z12get_group_idj";
static const char get_group_id[] = "_Z26__spirv_BuiltInWorkgroupIdi";
// static const char get_num_groups[] = "_Z14get_num_groupsj";
static const char get_num_groups[] = "_Z28__spirv_BuiltInNumWorkgroupsi";
// static const char get_sub_group_id[] = "_Z16get_sub_group_idv";
static const char get_sub_group_id[] = "_Z25__spirv_BuiltInSubgroupIdv";
// static const char get_sub_group_local_id[] = "_Z22get_sub_group_local_idv";
static const char get_sub_group_local_id[] =
    "_Z40__spirv_BuiltInSubgroupLocalInvocationIdv";
// static const char get_num_sub_groups[] = "_Z18get_num_sub_groupsv";
static const char get_num_sub_groups[] = "_Z27__spirv_BuiltInNumSubgroupsv";
// static const char get_sub_group_size[] = "_Z18get_sub_group_sizev";
static const char get_sub_group_size[] = "_Z27__spirv_BuiltInSubgroupSizev";

/*
_Z25__spirv_BuiltInGlobalSizei
_Z29__spirv_BuiltInGlobalLinearIdv
*/

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding VC intrinsics equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct GPUToSPIRPass : public impl::ConvertGPUToSPIRBase<GPUToSPIRPass> {
  using Base::Base;

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    // Request C wrapper emission.
    for (auto func : m.getOps<func::FuncOp>()) {
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(&getContext()));
    }

    // Collect kernels that need calling convention patch later
    llvm::SmallVector<StringRef, 4> kernels;
    for (auto gfunc : m.getOps<gpu::GPUFuncOp>()) {
      if (gfunc.isKernel())
        kernels.push_back(gfunc.getName());
    }

    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    options.useBarePtrCallConv = useBarePtrCallConv;

    // Apply in-dialect lowering. In-dialect lowering will replace
    // ops which need to be lowered further, which is not supported by a
    // single conversion pass.
    {
      RewritePatternSet patterns(m.getContext());
      // Convert gpu index ops to func.call to OCL builtins
      patterns.add<
          GPUIndexIntrinsicOpToOCLBuiltinLowering<gpu::ThreadIdOp,
                                                  get_local_id>,
          GPUIndexIntrinsicOpToOCLBuiltinLowering<gpu::BlockDimOp,
                                                  get_local_size>,
          GPUIndexIntrinsicOpToOCLBuiltinLowering<gpu::BlockIdOp, get_group_id>,
          GPUIndexIntrinsicOpToOCLBuiltinLowering<gpu::GridDimOp,
                                                  get_num_groups>,
          GPUIndexIntrinsicOpToOCLBuiltinLowering<gpu::GlobalIdOp,
                                                  get_global_id>,
          GPUSubgroupIndexIntrinsicOpToOCLBuiltinLowering<gpu::SubgroupIdOp,
                                                          get_sub_group_id>,
          GPUSubgroupIndexIntrinsicOpToOCLBuiltinLowering<
              gpu::LaneIdOp, get_sub_group_local_id>,
          GPUSubgroupIndexIntrinsicOpToOCLBuiltinLowering<gpu::NumSubgroupsOp,
                                                          get_num_sub_groups>,
          GPUSubgroupIndexIntrinsicOpToOCLBuiltinLowering<gpu::SubgroupSizeOp,
                                                          get_sub_group_size>>(
          patterns.getContext());
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
        return signalPassFailure();
    }
    LLVMTypeConverter converter(m.getContext(), options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) -> unsigned {
          switch (space) {
          case gpu::AddressSpace::Global:
            return 1;
          case gpu::AddressSpace::Workgroup:
            return 3;
          case gpu::AddressSpace::Private:
            return 0;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });
    populateSPIRVMemorySpaceAttributeConversions(
        converter, [](spirv::StorageClass space) -> unsigned {
          switch (space) {
          case spirv::StorageClass::CrossWorkgroup:
            return 1;
          case spirv::StorageClass::Workgroup:
            return 3;
          case spirv::StorageClass::Private:
            return 0;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });

    RewritePatternSet llvmPatterns(m.getContext());

    arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    llvmPatterns.add<GPUFuncOpLowering>(converter, 0, 0,
                                        builder.getStringAttr("spir_func"));
    llvmPatterns.add<GPUReturnOpLowering>(converter);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();

    // Post processing after lowering to LLVM

    // Apply Calling convention for spir
    // spir_kernel for kernel and spir_func for others.
    for (auto lfunc : m.getOps<LLVM::LLVMFuncOp>()) {
      bool isKernel = false;
      StringRef lfuncName = lfunc.getName();
      for (StringRef kname : kernels) {
        if (kname == lfuncName)
          isKernel = true;
      }
      if (isKernel) {
        lfunc.setCConv(LLVM::cconv::CConv::SPIR_KERNEL);
      } else {
        lfunc.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
      }
    }

    // Set gpu.module's data_layout and target attribute
    {
      m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
                 StringAttr::get(
                     m.getContext(),
                     "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
                     "f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:"
                     "64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:"
                     "256:256-v512:512:512-v1024:1024:1024"));
      m->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                 StringAttr::get(m.getContext(), "spir64-unknown-unknown"));
    }
  }
};

} // namespace
