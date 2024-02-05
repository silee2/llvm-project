include(FetchContent)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wglobal-constructors HAVE_GLOBAL_CONSTRUCTORS)

FetchContent_Declare(
  vc-intrinsics
  GIT_REPOSITORY https://github.com/intel/vc-intrinsics.git
  GIT_TAG master  # This will pull the latest changes from the master branch.
  # Don't auto-update on every build.
  UPDATE_DISCONNECTED 1
  # Allow manual updating with an explicit vc-intrinsics-update target.
  STEP_TARGETS update
  )
FetchContent_MakeAvailable(vc-intrinsics)
if (HAVE_GLOBAL_CONSTRUCTORS)
    target_compile_options(LLVMGenXIntrinsics PRIVATE -Wno-global-constructors)
endif()

add_mlir_conversion_library(MLIRGPUToVC
  GPUToVCPass.cpp

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
