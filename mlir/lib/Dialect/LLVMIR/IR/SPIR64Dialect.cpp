//===- SPIR64Dialect.cpp - SPIR64 IR Ops and Dialect registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types and operation details for the SPIR64 IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The SPIR64 dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/SPIR64Dialect.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace spir64;

#include "mlir/Dialect/LLVMIR/SPIR64OpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SPIR64Dialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

void SPIR64Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/SPIR64Ops.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/SPIR64OpsAttributes.cpp.inc"
      >();

  // Support unknown operations because not all SPIR64 operations are registered.
  allowUnknownOperations();
  declarePromisedInterface<SPIR64TargetAttr, gpu::TargetAttrInterface>();
}

LogicalResult SPIR64Dialect::verifyOperationAttribute(Operation *op,
                                                     NamedAttribute attr) {
  // Kernel function attribute should be attached to functions.
  if (attr.getName() == SPIR64Dialect::getKernelFuncAttrName()) {
    if (!isa<LLVM::LLVMFuncOp>(op)) {
      return op->emitError() << "'" << SPIR64Dialect::getKernelFuncAttrName()
                             << "' attribute attached to unexpected op";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SPIR64 target attribute.
//===----------------------------------------------------------------------===//
LogicalResult
SPIR64TargetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        int optLevel, StringRef triple, StringRef chip,
                        StringRef features,
                        DictionaryAttr flags, ArrayAttr files) {
  if (optLevel < 0 || optLevel > 3) {
    emitError() << "The optimization level must be a number between 0 and 3.";
    return failure();
  }
  if (triple.empty()) {
    emitError() << "The target triple cannot be empty.";
    return failure();
  }
  if (chip.empty()) {
    emitError() << "The target chip cannot be empty.";
    return failure();
  }
  if (files && !llvm::all_of(files, [](::mlir::Attribute attr) {
        return attr && mlir::isa<StringAttr>(attr);
      })) {
    emitError() << "All the elements in the `link` array must be strings.";
    return failure();
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/SPIR64Ops.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/SPIR64OpsAttributes.cpp.inc"
