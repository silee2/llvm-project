# LLVM GPU Offloading support
## Goal
Add minimal support for LLVM GPU offloading in upstream to enable downstream projects to
add support for GPUs without an upstream LLVM backend.

## Input
gpu.module containing device code with gpu.func. func body has gpu dialect ops.

## Overall flow
Step 1. Use upstream "attach llvm offload target" to attach vendor specific attributes.
Step 2: Use downstream conversion pass(es) that converts:
    - gpu ops that have no upstream lowering to LLVM
    - gpu ops that have upsteam lowering to LLVM but want to use vendor specific lowering.
    - other dialect ops that have upstream lowering to LLVM but want to use vendor specific lowering.
   To func.call to vendor specific intrinsic function.
Step 3: Use upstream GPUToLLVMOffload pass that converts gpu.func from Step 2 to llvm.func
Step 4: Use upstream gpu module to binary pass that converts llvm.func to gpu.binary
Step 5: Use upstream GPUToLLVM pass that lowers host code
Step 6: Use mlir-cpu-runner with downstream gpu execution engine wrapper that implements the common mgpuXXX API used by upstream. This execution engine accepts serialized LLVM bitcode as payload.

## Required changes
### Downstream:
1. Conversion pass for Step 2.
2. Execution engine that implements the common mgpuXXX API used by upstream for Step 6.

### Upstream:
1. llvm.offload_target attribute for Step 1.
2. attach llvm offload target passi for Step 1.
3. GPUToLLVMOffload pass for Step 3.
4. LLVMOffload target that only supports format=offload for Step 4.

## How to illustrate flow in upstream
Option 1: Write a test pass and a test execution engine targeting SPIR-V and OpenCL runtime and add some integration tests.
Option 2: Write an example with pass and execution engine targeting SPIR-V and OpenCL runtime and add some integration tests.

## Config LLVM
```sh
cmake -G Ninja -B build -S llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
    -DMLIR_ENABLE_SYCL_RUNNER=1 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## Run integration tests
```sh
cmake --build build --target check-mlir-integration-gpu-sycl
```
