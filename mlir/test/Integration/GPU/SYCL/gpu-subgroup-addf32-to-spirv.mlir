// RUN: mlir-opt %s -pass-pipeline='builtin.module(spirv-attach-target{ver=v1.0 caps=Addresses,Int64,Kernel},convert-gpu-to-spirv{use-64bit-index=true},gpu.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce)),func.func(llvm-request-c-wrappers),convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,gpu-to-llvm{use-bare-pointers-for-kernels=true},gpu-module-to-binary,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,reconcile-unrealized-casts)' \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_sycl_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module @add attributes {gpu.container_module} {
  memref.global "private" constant @__constant_2x4x4xf32_0 : memref<2x4x4xf32> = dense<[[[1.01, 1.02, 1.03, 1.04], [1.11, 1.12, 1.13, 1.14], [1.21, 1.22, 1.23, 1.24], [1.31, 1.32, 1.33, 1.34]], [[1.41, 1.42, 1.43, 1.44], [1.51, 1.52, 1.53, 1.54], [1.61, 1.62, 1.63, 1.64], [1.71, 1.72, 1.73, 1.74]]]>
  memref.global "private" constant @__constant_2x4x4xf32 : memref<2x4x4xf32> = dense<[[[2.01, 2.02, 2.03, 2.04], [2.11, 2.12, 2.13, 2.14], [2.21, 2.22, 2.23, 2.24], [2.31, 2.32, 2.33, 2.34]], [[2.41, 2.42, 2.43, 2.44], [2.51, 2.52, 2.53, 2.54], [2.61, 2.62, 2.63, 2.64], [2.71, 2.72, 2.73, 2.74]]]>
  func.func @main() {
    %0 = memref.get_global @__constant_2x4x4xf32 : memref<2x4x4xf32>
    %1 = memref.get_global @__constant_2x4x4xf32_0 : memref<2x4x4xf32>
    %2 = call @test(%0, %1) : (memref<2x4x4xf32>, memref<2x4x4xf32>) -> memref<2x4x4xf32>
    %cast = memref.cast %2 : memref<2x4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @test(%arg0: memref<2x4x4xf32>, %arg1: memref<2x4x4xf32>) -> memref<2x4x4xf32> {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %mem = gpu.alloc host_shared () : memref<2x4x4xf32>
    memref.copy %arg1, %mem : memref<2x4x4xf32> to memref<2x4x4xf32>
    %memref_0 = gpu.alloc host_shared () : memref<2x4x4xf32>
    memref.copy %arg0, %memref_0 : memref<2x4x4xf32> to memref<2x4x4xf32>
    %memref_2 = gpu.alloc host_shared () : memref<2x4x4xf32>
    %2 = gpu.wait async
    %3 = gpu.launch_func async [%2] @test_kernel::@test_kernel blocks in (%c2, %c1, %c1) threads in (%c4, %c4, %c1) args(%memref_0 : memref<2x4x4xf32>, %mem : memref<2x4x4xf32>, %memref_2 : memref<2x4x4xf32>)
    gpu.wait [%3]
    %alloc = memref.alloc() : memref<2x4x4xf32>
    memref.copy %memref_2, %alloc : memref<2x4x4xf32> to memref<2x4x4xf32>
    %4 = gpu.wait async
    %5 = gpu.dealloc async [%4] %memref_2 : memref<2x4x4xf32>
    %6 = gpu.dealloc async [%5] %memref_0 : memref<2x4x4xf32>
    %7 = gpu.dealloc async [%6] %mem : memref<2x4x4xf32>
    gpu.wait [%7]
    return %alloc : memref<2x4x4xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<2x4x4xf32>, %arg1: memref<2x4x4xf32>, %arg2: memref<2x4x4xf32>) kernel attributes {gpu.known_block_size = array<i32: 4, 4, 1>, gpu.known_grid_size = array<i32: 2, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c4 = arith.constant 4 : index
      %0 = gpu.block_id  x
      %1 = gpu.subgroup_id : index
      %2 = gpu.subgroup_size : index
      %3 = gpu.lane_id
      %4 = arith.muli %1, %2 : index
      %5 = arith.addi %3, %4 : index
      %6 = arith.divui %5, %c4 : index
      %7 = arith.remui %5, %c4 : index
      %8 = memref.load %arg0[%0, %6, %7] : memref<2x4x4xf32>
      %9 = memref.load %arg1[%0, %6, %7] : memref<2x4x4xf32>
      %10 = arith.addf %8, %9 : f32
      memref.store %10, %arg2[%0, %6, %7] : memref<2x4x4xf32>
      gpu.return
    }
  }
  // CHECK: [3.02,    3.04,    3.06,    3.08]
  // CHECK: [3.22,    3.24,    3.26,    3.28]
  // CHECK: [3.42,    3.44,    3.46,    3.48]
  // CHECK: [3.62,    3.64,    3.66,    3.68]
  // CHECK: [3.82,    3.84,    3.86,    3.88]
  // CHECK: [4.02,    4.04,    4.06,    4.08]
  // CHECK: [4.22,    4.24,    4.26,    4.28]
  // CHECK: [4.42,    4.44,    4.46,    4.48]
}
