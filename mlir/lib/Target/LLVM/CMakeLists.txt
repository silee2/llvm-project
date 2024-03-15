add_mlir_library(MLIRTargetLLVM
  ModuleToObject.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVM

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  IPO
  IRReader
  Linker
  MC
  Passes
  Support
  Target
  LINK_LIBS PUBLIC
  MLIRExecutionEngineUtils
  MLIRTargetLLVMIRExport
)

if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(NVPTX_LIBS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

add_mlir_dialect_library(MLIRNVVMTarget
  NVVM/Target.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  LINK_COMPONENTS
  ${NVPTX_LIBS}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRExecutionEngineUtils
  MLIRSupport
  MLIRGPUDialect
  MLIRTargetLLVM
  MLIRNVVMToLLVMIRTranslation
  )

if(MLIR_ENABLE_CUDA_CONVERSIONS)
  # Find the CUDA toolkit.
  find_package(CUDAToolkit)

  if(CUDAToolkit_FOUND)
    # Get the CUDA toolkit path. The path is needed for detecting `libdevice.bc`.
    # These extra steps are needed because of a bug on CMake.
    # See: https://gitlab.kitware.com/cmake/cmake/-/issues/24858
    # TODO: Bump the MLIR CMake version to 3.26.4 and switch to
    # ${CUDAToolkit_LIBRARY_ROOT}
    if(NOT DEFINED ${CUDAToolkit_LIBRARY_ROOT})
      get_filename_component(MLIR_CUDAToolkit_ROOT ${CUDAToolkit_BIN_DIR}
                             DIRECTORY ABSOLUTE)
    else()
      set(MLIR_CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_ROOT})
    endif()

    # Add the `nvptxcompiler` library.
    if(MLIR_ENABLE_NVPTXCOMPILER)
      # Find the `nvptxcompiler` library.
      # TODO: Bump the MLIR CMake version to 3.25 and use `CUDA::nvptxcompiler_static`.
      find_library(MLIR_NVPTXCOMPILER_LIB nvptxcompiler_static
                  PATHS ${CUDAToolkit_LIBRARY_DIR} NO_DEFAULT_PATH)

      # Fail if `nvptxcompiler_static` couldn't be found.
      if(MLIR_NVPTXCOMPILER_LIB STREQUAL "MLIR_NVPTXCOMPILER_LIB-NOTFOUND")
        message(FATAL_ERROR
                "Requested using the `nvptxcompiler` library backend but it couldn't be found.")
      endif()

      # Link against `nvptxcompiler_static`. TODO: use `CUDA::nvptxcompiler_static`.
      target_link_libraries(MLIRNVVMTarget PRIVATE ${MLIR_NVPTXCOMPILER_LIB})
      target_include_directories(obj.MLIRNVVMTarget PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    endif()
  else()
    # Fail if `MLIR_ENABLE_NVPTXCOMPILER` is enabled and the toolkit couldn't be found.
    if(MLIR_ENABLE_NVPTXCOMPILER)
      message(FATAL_ERROR
              "Requested using the `nvptxcompiler` library backend but it couldn't be found.")
    endif()
  endif()
  message(VERBOSE "MLIR default CUDA toolkit path: ${MLIR_CUDAToolkit_ROOT}")

  # Define the `CUDAToolkit` path.
  target_compile_definitions(obj.MLIRNVVMTarget
    PRIVATE
    MLIR_NVPTXCOMPILER_ENABLED=${MLIR_ENABLE_NVPTXCOMPILER}
    __DEFAULT_CUDATOOLKIT_PATH__="${MLIR_CUDAToolkit_ROOT}"
  )
endif()

if (MLIR_ENABLE_ROCM_CONVERSIONS)
  set(AMDGPU_LIBS
    AMDGPUAsmParser
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
  )
endif()

add_mlir_dialect_library(MLIRROCDLTarget
  ROCDL/Target.cpp

  LINK_COMPONENTS
  MCParser
  ${AMDGPU_LIBS}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRExecutionEngineUtils
  MLIRSupport
  MLIRGPUDialect
  MLIRTargetLLVM
  MLIRROCDLToLLVMIRTranslation
  )

if(MLIR_ENABLE_ROCM_CONVERSIONS)
  if (NOT ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building mlir with ROCm support requires the AMDGPU backend")
  endif()

  if (DEFINED ROCM_PATH)
    set(DEFAULT_ROCM_PATH "${ROCM_PATH}" CACHE PATH "Fallback path to search for ROCm installs")
  elseif(DEFINED ENV{ROCM_PATH})
    set(DEFAULT_ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Fallback path to search for ROCm installs")
  else()
    set(DEFAULT_ROCM_PATH "/opt/rocm" CACHE PATH "Fallback path to search for ROCm installs")
  endif()
  message(VERBOSE "MLIR Default ROCM toolkit path: ${DEFAULT_ROCM_PATH}")

  target_compile_definitions(obj.MLIRROCDLTarget
    PRIVATE
    __DEFAULT_ROCM_PATH__="${DEFAULT_ROCM_PATH}"
  )
endif()

if (MLIR_ENABLE_SPIRV_CONVERSIONS)
  set(SPIRV_LIBS
    SPIRVCodeGen
    SPIRVDesc
    SPIRVInfo
  )
  if (MLIR_ENABLE_SPIRV_LLVM_TRANSLATOR)
    # Two cmake targets LLVMSPIRVLib, SPIRV-Tools-static are needed
    # and source is fetched if missing.
    # If any of them needs to be fetched, target SPIRV-Headers is needed
    # and fetched.

    include(FetchContent)
    include(CheckCXXCompilerFlag)

    check_cxx_compiler_flag(-Werror HAVE_ERROR)

    if((NOT TARGET SPIRV-Tools-static) OR (NOT TARGET LLVMSPIRVLib))
      message(STATUS "Fetching SPIRV Headers")
      FetchContent_Declare(
        spirv-headers
        GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
        GIT_TAG vulkan-sdk-1.3.275.0
      )
      FetchContent_MakeAvailable(spirv-headers)
    endif()

    if(NOT TARGET SPIRV-Tools-static)
      message(STATUS "Fetching SPIRV Tools")
      set(SPIRV_SKIP_TESTS ON)
      set(SPIRV-Headers_SOURCE_DIR ${spirv-headers_SOURCE_DIR})
      FetchContent_Declare(
        spirv-tools
        GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
        GIT_TAG vulkan-sdk-1.3.275.0
      )
      FetchContent_MakeAvailable(spirv-tools)
      if (HAVE_ERROR)
          target_compile_options(SPIRV-Tools-static PRIVATE -Wno-error)
      endif()
    endif()

    if(NOT TARGET LLVMSPIRVLib)
      message(STATUS "Fetching SPIRV LLVM Translator")
      set(LLVM_SPIRV_INCLUDE_TESTS OFF)
      set(LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR ${spirv-headers_SOURCE_DIR})
      check_cxx_compiler_flag(-Wglobal-constructors HAVE_GLOBAL_CONSTRUCTORS)
      FetchContent_Declare(
        spirv-llvm-translator
        GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
        GIT_TAG 2cd8b788edc53ed144eb8644781b37a741071b09
      )
      FetchContent_MakeAvailable(spirv-llvm-translator)
      if (HAVE_GLOBAL_CONSTRUCTORS)
        target_compile_options(LLVMSPIRVLib PRIVATE -Wno-global-constructors)
      endif()
      if (HAVE_ERROR)
        target_compile_options(LLVMSPIRVLib PRIVATE -Wno-error)
      endif()
    endif()
  endif()
endif()

add_mlir_dialect_library(MLIRSPIRTarget
  SPIR/Target.cpp

  LINK_COMPONENTS
  MCParser
  ${SPIRV_LIBS}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRExecutionEngineUtils
  MLIRSupport
  MLIRGPUDialect
  MLIRTargetLLVM
  MLIRSPIRToLLVMIRTranslation
  )

if (MLIR_ENABLE_SPIRV_LLVM_TRANSLATOR)
  get_target_property(_src_dir LLVMSPIRVLib SOURCE_DIR)
  target_include_directories(MLIRSPIRTarget
  PRIVATE
    ${_src_dir}/../../include
  )
  target_link_libraries(MLIRSPIRTarget
  PRIVATE
    SPIRV-Tools-static
    LLVMSPIRVLib
  )
endif()
