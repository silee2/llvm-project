# CMake find_package() module for SYCL Runtime
#
# Example usage:
#
# find_package(SyclRuntime)
#
# If successful, the following variables will be defined:
# SyclRuntime_FOUND
#

if (TARGET SyclRuntime::SyclRuntime)
    # If the target is already defined, we assume that the package has been found.
    set(SyclRuntime_FOUND TRUE)
    return()
endif()

include(FindPackageHandleStandardArgs)

# if user provide CMake variable SyclRuntime_ROOT, that will have the highest priority for search.
# if the environment variable CMPLR_ROOT is set, we will use it as a hint to find the SYCL runtime.
if(DEFINED ENV{CMPLR_ROOT})
    get_filename_component(ONEAPI_VER "$ENV{CMPLR_ROOT}" NAME)
    # Need at least 2024.1 for experimental SPIR-V compiler support
    if(ONEAPI_VER VERSION_LESS 2024.1)
        # set hint to null path as we don't have a valid SYCL runtime with DPC++ compiler version < 2024.1
        set(SyclRuntime_HINT "")
    else()
        set(SyclRuntime_HINT "$ENV{CMPLR_ROOT}")
    endif()
else ()
    # don't have a valid SYCL runtime, so we set the hint to an empty string
    set(SyclRuntime_HINT "")
endif()

find_library(SyclRuntime_LIBRARY
    NAMES sycl
    HINTS ${SyclRuntime_HINT}
    NO_CACHE
    )

if(SyclRuntime_LIBRARY)
    set(SyclRuntime_FOUND TRUE)
    find_package_message(SyclRuntime "Found SyclRuntime: ${SyclRuntime_LIBRARY}" "")
    add_library(SyclRuntime::SyclRuntime INTERFACE IMPORTED)
    set_target_properties(SyclRuntime::SyclRuntime
        PROPERTIES INTERFACE_LINK_LIBRARIES "${SyclRuntime_LIBRARY}"
    )
    cmake_path(GET SyclRuntime_LIBRARY PARENT_PATH SyclRuntime_LIBRARY_DIR)
    cmake_path(GET SyclRuntime_LIBRARY_DIR PARENT_PATH SyclRuntime_BASE_DIR)

    list(APPEND SyclRuntime_INCLUDE_DIRS "${SyclRuntime_BASE_DIR}/include")
    list(APPEND SyclRuntime_INCLUDE_DIRS "${SyclRuntime_BASE_DIR}/include/sycl")

    set_target_properties(SyclRuntime::SyclRuntime
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SyclRuntime_INCLUDE_DIRS}"
    )
else()
    set(SyclRuntime_FOUND FALSE)
    find_package_message(SyclRuntime "Could not find SyclRuntime" "")
endif()

find_package_handle_standard_args(SyclRuntime
    REQUIRED_VARS
        SyclRuntime_FOUND
    HANDLE_COMPONENTS
)

