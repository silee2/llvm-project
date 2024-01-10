//===-- Passes.td - XeGPU pass definition file --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines passes/transformations of the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XeGPU_PASSES_TD_INCLUDED_
#define _XeGPU_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// XeGPUToSPIRVWithVCIntrinsics pass
//===----------------------------------------------------------------------===//

// def XeGPUToSPIRVWithVCIntrinsics: Pass<"xegpu-spirvWithVCIntrinsics", "::mlir::func::FuncOp"> {
//   let summary = "Add a pass for lowering XeGPU dialect ops to spirv with VC intrinsics";
//   let description = [{
//
//     #### Input invariant
//
//
//     #### Output IR
//
//   }];
//   let constructor = "::mlir::createXeGPUToSPIRVWithVCIntrinsicsPass()";
//   let dependentDialects = ["::mlir::xegpu::XeGPUDialect"];
//   let options = [];
// }

//===----------------------------------------------------------------------===//
// XeGPUToSPIRVWithJointMatrix pass
//===----------------------------------------------------------------------===//

// def XeGPUToSPIRVWithJointMatrix: Pass<"xegpu-spirvWithJointMatrix", "::mlir::func::FuncOp"> {
//   let summary = "Add a pass for lowering XeGPU dialect ops to spirv with Joint matrix extension";
//   let description = [{
//
//     #### Input invariant
//
//
//     #### Output IR
//
//   }];
//   let constructor = "::mlir::createXeGPUToSPIRVWithJointMatrixPass()";
//   let dependentDialects = ["::mlir::xegpu::XeGPUDialect"];
//   let options = [];
// }

#endif // _XeGPU_PASSES_TD_INCLUDED_
