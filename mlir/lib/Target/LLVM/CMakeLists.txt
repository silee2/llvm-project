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
    # Build translator from source
    if(NOT TARGET LLVMSPIRVLib)
    message(STATUS "Need to build SPIRV LLVM Translator")
    include(FetchContent)

    # check if spirv-tools need to be built from source
    set(NEED_TO_BUILD_SPIRV_TOOLS TRUE)
    include(FindPkgConfig)
    # first try locating SPIRV-Tools via pkgconfig (the old way)
    pkg_search_module(SPIRV_TOOLS SPIRV-Tools)
    if (SPIRV_TOOLS_FOUND)
      set(NEED_TO_BUILD_SPIRV_TOOLS FALSE)
      message(STATUS "Found SPIRV Tools old way")
    else()
      # then try locating SPIRV-Tools via cmake (the new way)
      find_package(SPIRV-Tools)
      find_package(SPIRV-Tools-tools)
      if (SPIRV-Tools_FOUND AND SPIRV-Tools-tools_FOUND)
        set(NEED_TO_BUILD_SPIRV_TOOLS FALSE)
        message(STATUS "Found SPIRV Tools new way")
      endif()
    endif()

    if(NEED_TO_BUILD_SPIRV_TOOLS)
    message(STATUS "Need to build SPIRV Tools")
    set(SPIRV_SKIP_TESTS ON)
    FetchContent_Declare(
      spirv-tools
      GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
      GIT_TAG main  # This will pull the latest changes from the main branch.
      DOWNLOAD_COMMAND git clone https://github.com/KhronosGroup/SPIRV-Tools.git ${FETCHCONTENT_BASE_DIR}/spirv-tools-src &&
      cd ${FETCHCONTENT_BASE_DIR}/spirv-tools-src &&
        ${Python3_EXECUTABLE} utils/git-sync-deps
      UPDATE_COMMAND cd ${FETCHCONTENT_BASE_DIR}/spirv-tools-src &&
        git pull origin main &&
        ${Python3_EXECUTABLE} utils/git-sync-deps
      # Don't auto-update on every build.
      UPDATE_DISCONNECTED 1
      # Allow manual updating with an explicit SPIRVTools-update target.
      STEP_TARGETS update
    )
    FetchContent_MakeAvailable(spirv-tools)
    message(STATUS "SPIRV Tools binary dir: ${spirv-tools_BINARY_DIR}")
    message(STATUS "SPIRV Tools source dir: ${spirv-tools_SOURCE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${spirv-tools_BINARY_DIR}")
    set(LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR ${spirv-tools_SOURCE_DIR}/external/spirv-headers)
    endif()

    set(LLVM_SPIRV_INCLUDE_TESTS OFF)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-Wglobal-constructors HAVE_GLOBAL_CONSTRUCTORS)
    FetchContent_Declare(
      spirv-llvm-translator
      GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
      GIT_TAG main  # This will pull the latest changes from the main branch.
    )
    FetchContent_MakeAvailable(spirv-llvm-translator)
    if (HAVE_GLOBAL_CONSTRUCTORS)
      target_compile_options(LLVMSPIRVLib PRIVATE -Wno-global-constructors)
    endif()

    set(SPIRV_LLVM_LIBS
      SPIRVLib
    )
    endif()
  endif()
endif()

add_mlir_dialect_library(MLIRSPIR64Target
  SPIR64/Target.cpp

  LINK_COMPONENTS
  MCParser
  ${SPIRV_LIBS}
  ${SPIRV_LLVM_LIBS}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRExecutionEngineUtils
  MLIRSupport
  MLIRGPUDialect
  MLIRTargetLLVM
  MLIRSPIR64ToLLVMIRTranslation
  )

if (MLIR_ENABLE_SPIRV_LLVM_TRANSLATOR)
  target_include_directories(MLIRSPIR64Target
  PRIVATE
    ${spirv-llvm-translator_SOURCE_DIR}/include
    ${spirv-tools_SOURCE_DIR}/include
  )
  target_link_libraries(MLIRSPIR64Target
  PRIVATE
    ${SPIRV_TOOLS_LDFLAGS}
  )
endif()

