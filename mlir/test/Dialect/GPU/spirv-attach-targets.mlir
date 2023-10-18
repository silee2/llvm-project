// RUN: mlir-opt %s --spirv-attach-target='module=spirv.* addressing=Physical64 memory=OpenCL' | FileCheck %s
// RUN: mlir-opt %s --spirv-attach-target='module=spirv.*' | FileCheck %s --check-prefix=CHECK_DEF
// RUN: mlir-opt %s --spirv-attach-target='module=spirv.* addressing=Logical memory=GLSL450' | FileCheck %s --check-prefix=CHECK_GLSL

module attributes {gpu.container_module} {
// Verify the target with addressing model and memory model same as default options. Default values do not print.
// CHECK: @spirv_module_1 [#spirv.target] {
// Verify the target with no addressing model and memory model. They will pick up defualt value.
// CHECK_DEF: @spirv_module_1 [#spirv.target] {
// Verify the target with non default options.
// CHECK_GLSL: @spirv_module_1 [#spirv.target<addressing = "Logical", memory = "GLSL450">] {
gpu.module @spirv_module_1 {
}
}
