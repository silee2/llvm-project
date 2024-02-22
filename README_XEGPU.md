# XeGPU lowering

## Config
```
cmake -G Ninja -B build -S llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
    -DMLIR_ENABLE_SPIRV_LLVM_TRANSLATOR=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## Run integration tests
```
cmake --build build --target check-mlir-integration-gpu-sycl
```
