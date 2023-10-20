// RUN: mlir-opt -convert-gpu-to-spirv='use-64bit-index=true use-opencl=true' %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Kernel, Int64, Addresses], []>, api=OpenCL, #spirv.resource_limits<>>
} {

  // CHECK-LABEL: spirv.module @{{.*}} Physical64 OpenCL
  gpu.module @kernels {
    // CHECK-LABEL: spirv.func @load_store_kernel
    gpu.func @load_store_kernel(%arg0: memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>, %arg1: memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>, %arg2: memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) kernel
      attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 16, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      gpu.return
    }
    // CHECK-LABEL: func.func @load_store_kernel
    // CHECK-SAME: attributes {gpu.kernel}
  }
}
