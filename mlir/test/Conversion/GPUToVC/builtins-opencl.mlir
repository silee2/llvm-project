// RUN: mlir-opt -split-input-file -convert-gpu-to-vc %s -o - | FileCheck %s

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
    gpu.func @empty_kernel() kernel
      attributes {} {
      gpu.return
    }
  }
}

