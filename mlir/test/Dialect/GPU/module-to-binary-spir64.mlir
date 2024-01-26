// REQUIRES: host-supports-spirv
// RUN: mlir-opt %s --gpu-module-to-binary="format=llvm" | FileCheck %s
// RUN: mlir-opt %s --gpu-module-to-binary="format=isa" | FileCheck %s -check-prefix=CHECK-ISA

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

  // CHECK-LABEL:gpu.binary @kernel_module3 <#gpu.select_object<1 : i64>>
  // CHECK:[#gpu.object<#spir64.target<chip = "XeHPC">, offload = "{{.*}}">, #gpu.object<#spir64.target<chip = "XeHPG">, offload = "{{.*}}">]
  gpu.module @kernel_module3 <#gpu.select_object<1>> [
      #spir64.target<chip = "XeHPC">,
      #spir64.target<chip = "XeHPG">] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
