// REQUIRES: host-supports-spirv
// RUN: mlir-opt %s --gpu-module-to-binary="format=llvm" | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL:gpu.binary @kernel_module1
  // CHECK:[#gpu.object<#spir64.target<chip = "XeHPC">, offload = "{{.*}}">]
  gpu.module @kernel_module1 [#spir64.target<chip = "XeHPC">] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
