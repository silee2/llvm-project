add_subdirectory(Deserialization)
add_subdirectory(Serialization)

set(LLVM_OPTIONAL_SOURCES
  SPIRVBinaryUtils.cpp
  TranslateRegistration.cpp
  Target.cpp
  )

add_mlir_translation_library(MLIRSPIRVBinaryUtils
  SPIRVBinaryUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRVDialect
  MLIRSupport
  )

add_mlir_translation_library(MLIRSPIRVTranslateRegistration
  TranslateRegistration.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRVDialect
  MLIRSPIRVSerialization
  MLIRSPIRVDeserialization
  MLIRSupport
  MLIRTranslateLib
  )

add_mlir_dialect_library(MLIRSPIRVTarget
  Target.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRGPUDialect
  MLIRSPIRVDialect
  MLIRSPIRVSerialization
  MLIRSupport
  )

if (MLIR_ENABLE_SPIRV_LLVM_TRANSLATOR)
  target_link_libraries(MLIRSPIRVTarget
  PRIVATE
    SPIRV-Tools-static
  )
endif()
