# LLVM GPU Offloading support
## TODO: Update this document for LLVM GPU Offloading support
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
