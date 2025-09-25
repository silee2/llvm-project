add_mlir_conversion_library(MLIRXeVMToLLVMOCL
  XeVMToLLVMOCL.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/XeVMToLLVMOCL

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRXeVMDialect
  MLIRPass
  MLIRTransforms
)
