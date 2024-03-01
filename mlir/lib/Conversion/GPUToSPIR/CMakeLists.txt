add_mlir_conversion_library(MLIRGPUToSPIR
    GPUToSPIRPass.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  GenXIntrinsics

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIRNVVMDialect
  MLIRPass
  MLIRTransformUtils
  )
