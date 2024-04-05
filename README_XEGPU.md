# XeGPU lowering
## Prerequisite
DPC++ compiler (https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html)

## Setup SYCL runtime
Run setvars.sh from DPC++
```sh
/opt/intel/oneapi/setvars.sh
```

## Config LLVM
Following confguration will build and use Khronos SPIR-V LLVM translator and use it
```sh
cmake -G Ninja -B build -S llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
    -DMLIR_ENABLE_SPIRV_LLVM_TRANSLATOR=1 \
    -DMLIR_ENABLE_SYCL_RUNNER=1 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```
To use LLVM's own SPIR-V backend, change MLIR_ENABLE_SPIRV_LLVM_TRANSLATOR option to OFF
```sh
    -DMLIR_ENABLE_SPIRV_LLVM_TRANSLATOR=0
```


## Run integration tests
```sh
cmake --build build --target check-mlir-integration-gpu-sycl
```

## Using with IMEX
```sh
git clone https://github.com/intel/mlir-extensions.git
git clone -b xegpu_lowering https://github.com/silee2/llvm-project.git
cd llvm-project
git checkout `cat ../mlir-extensions/build_tools/llvm_version.txt`
git apply ../mlir-extensions/build_tools/patches/*
cmake -G Ninja -B build -S llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
    -DMLIR_ENABLE_SPIRV_LLVM_TRANSLATOR=1 \
    -DMLIR_ENABLE_SYCL_RUNNER=1 \
    -DLLVM_EXTERNAL_PROJECTS="Imex" \
    -DLLVM_EXTERNAL_IMEX_SOURCE_DIR=../mlir-extensions \
    -DIMEX_CHECK_LLVM_VERSION=OFF \
    <Other IMEX CMake Options ...>
```
To run SYCL integration test:
```sh
cmake --build build --target check-mlir-integration-gpu-sycl
```
Lowering pipeline from GPU dialect:
```sh
./build/bin/mlir-opt <path-to-test> -pass-pipeline='builtin.module(spir-attach-target{module=test.* chip=XeHPC ver=v1.0 caps=Kernel},func.func(gpu-async-region),gpu.module(map-memref-spirv-storage-class{client-api=opencl},convert-gpu-to-spir),func.func(llvm-request-c-wrappers),convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,gpu-to-llvm{use-bare-pointers-for-kernels=true},gpu-module-to-binary{format=bin},expand-strided-metadata,lower-affine,finalize-memref-to-llvm,reconcile-unrealized-casts)' | \
./build/bin/mlir-cpu-runner    --shared-libs=./build/lib/libmlir_sycl_runtime.so  \
    --shared-libs=./build/lib/libmlir_runner_utils.so  \
    --entry-point-result=void
```

## Dump SPIR-V text/assembly
```sh
./build/bin/mlir-opt <path-to-test> -pass-pipeline='builtin.module(spir-attach-target{module=test.* chip=XeHPC ver=v1.0 caps=Kernel},func.func(gpu-async-region),gpu.module(map-memref-spirv-storage-class{client-api=opencl},convert-gpu-to-spir),func.func(llvm-request-c-wrappers),convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,gpu-to-llvm{use-bare-pointers-for-kernels=true},gpu-module-to-binary{format=isa},expand-strided-metadata,lower-affine,finalize-memref-to-llvm,reconcile-unrealized-casts)' --debug-only=serialize-spir-to-isa > /dev/null 2> <somename>.spt
```

## Implementaion Files Of Interest
### LLVM to SPIR-V translation (serializer)
```
mlir/lib/Target/LLVM/SPIR/Target.cpp
```
### GPUToSPIR Pass (GPU dialect to LLVM dialect conversion)
```
mlir/lib/Conversion/GPUToSPIR/GPUToSPIRPass.cpp
```

## TODO:
### GPUToSPIR
### Translation

## Design workflow

### Input
gpu.module containing device code with gpu.func. func body has gpu dialect ops and may contain downstream dialect ops as well.

### Overall flow
Step 1. Use upstream "attach spir target" to attach spir.target with options for SPIR-V serialization.
Step 2: Use downstream conversion pass(es) that converts:
    - gpu ops that have no upstream lowering to LLVM
    - gpu ops that have upsteam lowering to LLVM but want to use vendor specific lowering.
    - other upstream dialect ops that have upstream lowering to LLVM but want to use vendor specific lowering.
    - downstream dialect that does not have a lowering to LLVM.
   To func.call to vendor specific intrinsic function.
Step 3: Use upstream GPUToSPIR pass that converts gpu.func from Step 2 to llvm.func. Note that This pass handles gpu index op lowering.
Step 4: Use upstream gpu module to binary pass that converts llvm.func to gpu.binary
Step 5: Use upstream GPUToLLVM pass that lowers host code
Step 6: Use mlir-cpu-runner with upstream SYCL wrapper.

Note that only Step 2: requires downstream a downstream component. All other steps are upstream.

### Upstream changes:
1. spir dialect with spir.target as the only op for Step 1.
2. attach spir target pass for Step 1.
3. GPUToSPIR pass for Step 3.
4. mlir/lib/Target/LLVM/SPIR/Target.cpp for Step 4.
5. llvm/lib/Target/SPIRV updates to support SPIR-V serialization for Step 4.
