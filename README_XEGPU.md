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


