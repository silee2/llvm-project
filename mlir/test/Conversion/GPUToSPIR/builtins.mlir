// RUN: mlir-opt -split-input-file -convert-gpu-to-spir %s -o - | FileCheck %s

module attributes {
  gpu.container_module
} {
  gpu.module @test_kernel {
    // CHECK: llvm.func spir_funccc @_Z26__spirv_BuiltInWorkgroupIdi(i32) -> i64 attributes {sym_visibility = "private"}
    // CHECK: llvm.func spir_kernelcc @test_kernel
    gpu.func @test_kernel(%arg0: memref<2x2x2xf32>, %arg1: memref<2x2x2xf32>, %arg2: memref<2x2x2xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 2, 2, 2>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      // CHECK-DAG: [[v0:%.*]] = llvm.mlir.constant(0 : i32) : i32
      // CHECK-DAG: [[v1:%.*]] = llvm.mlir.constant(1 : i32) : i32
      // CHECK-DAG: [[v2:%.*]] = llvm.mlir.constant(2 : i32) : i32
      // CHECK-DAG: [[idx0:%.*]] = llvm.call @_Z26__spirv_BuiltInWorkgroupIdi([[v0]]) : (i32) -> i64
      // CHECK-DAG: [[idx1:%.*]] = llvm.call @_Z26__spirv_BuiltInWorkgroupIdi([[v1]]) : (i32) -> i64
      // CHECK-DAG: [[idx2:%.*]] = llvm.call @_Z26__spirv_BuiltInWorkgroupIdi([[v2]]) : (i32) -> i64
      %3 = memref.load %arg0[%0, %1, %2] : memref<2x2x2xf32>
      %4 = memref.load %arg1[%0, %1, %2] : memref<2x2x2xf32>
      %5 = arith.addf %3, %4 : f32
      memref.store %5, %arg2[%0, %1, %2] : memref<2x2x2xf32>
      gpu.return
    }
  }
}

