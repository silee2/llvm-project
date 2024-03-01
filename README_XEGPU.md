# XeGPU lowering
## Prerequisite
DPC++ compiler (https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html)

## Setup SYCL runtime
Run setvars.sh from DPC++
```
/opt/intel/oneapi/setvars.sh
```

## Config LLVM
Following confguration will build and use Khronos SPIR-V LLVM translator and use it
```
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
```
    -DMLIR_ENABLE_SPIRV_LLVM_TRANSLATOR=0
```


## Run integration tests
```
cmake --build build --target check-mlir-integration-gpu-sycl
```
