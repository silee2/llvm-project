add_mlir_translation_library(MLIRSPIR64ToLLVMIRTranslation
  SPIR64ToLLVMIRTranslation.cpp

  DEPENDS
  MLIRSPIR64ConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSPIR64Dialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
