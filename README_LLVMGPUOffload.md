# LLVM GPU Offloading support
## TODO: Update this document for LLVM GPU Offloading support
## Purpose
Add minimal support for LLVM GPU offloading to enable downstream projects to
add support for GPUs without an upstream LLVM backend.

## Overall flow
1. Input with gpu.func using gpu dialect.
2. Upstream "attach llvm offload target" to attach vendor specific attributes.
3. Downstream conversion pass that converts part of gpu.func that lowers to func.calls to vendor specific intrinsic function.
4. Upstream GPUToLLVMOffload pass that converts gpu.func to llvm.func
5. Upstream gpu module to binary pass that converts llvm.func to gpu.binary
6. Upstream GPUToLLVM pass that lowers host side code
7. Downstream gpu execution engine that implements the common mgpuXXX API used by upstream. This execution engine accepts serialized LLVM bitcode as payload.

In summary, downstream needs to implement the following:
1. Conversion pass that converts gpu.func to vendor specific intrinsic function.
2. Execution engine that implements the common mgpuXXX API used by upstream.

Upstream needs to implement the following:
1. llvm.offload_target attribute
2. attach llvm offload target pass
3. GPUToLLVMOffload pass
4. LLVMOffload target that only supports format=offload

## How to demonstrate flow in upstream
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
