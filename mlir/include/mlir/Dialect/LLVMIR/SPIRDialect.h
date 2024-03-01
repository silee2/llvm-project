//===- SPIRDialect.h - MLIR SPIR IR dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SPIR dialect in MLIR, including Target attribute
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_SPIRDIALECT_H
#define MLIR_DIALECT_LLVMIR_SPIRDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

///// Ops /////
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/SPIROpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/SPIROps.h.inc"

#include "mlir/Dialect/LLVMIR/SPIROpsDialect.h.inc"

#endif /* MLIR_DIALECT_LLVMIR_SPIRDIALECT_H */
