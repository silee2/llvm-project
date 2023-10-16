// RUN: mlir-opt %s --spirv-attach-target='module=spirv.* amod=Physical64 mmod=OpenCL' | FileCheck %s
// RUN: mlir-opt %s --spirv-attach-target='module=spirv.* amod=Logical mmod=GLSL450' | FileCheck %s --check-prefix=CHECK_GLSL

module attributes {gpu.container_module} {
// Verify the target with default options.
// CHECK: @spirv_module_1 [#spirv.target] {
// Verify the target with non default options.
// CHECK_GLSL: @spirv_module_1 [#spirv.target<amod = "Logical", mmod = "GLSL450">] {
gpu.module @spirv_module_1 {
}
}
