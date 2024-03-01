add_mlir_translation_library(MLIRSPIRToLLVMIRTranslation
  SPIRToLLVMIRTranslation.cpp

  DEPENDS
  MLIRSPIRConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSPIRDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
