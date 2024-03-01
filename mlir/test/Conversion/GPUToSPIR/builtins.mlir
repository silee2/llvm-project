// RUN: mlir-opt -split-input-file -convert-gpu-to-spir %s -o - | FileCheck %s

module attributes {
  gpu.container_module
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@empty_kernel
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  gpu.module @kernels {
    // CHECK: llvm.func spir_kernelcc @empty_kernel
    gpu.func @empty_kernel() kernel
      attributes {} {
      %0 = gpu.thread_id y
      %1 = gpu.block_id x
      %2 = gpu.grid_dim z
      %3 = gpu.block_dim y
      gpu.return
    }
  }
}

