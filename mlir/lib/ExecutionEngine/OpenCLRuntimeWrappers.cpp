//===- OpenCLRuntimeWrappers.cpp - MLIR OpenCL wrapper library ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the OpenCL runtime library.
//
//===----------------------------------------------------------------------===//

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#ifdef _WIN32
#define OPENCL_RUNTIME_EXPORT __declspec(dllexport)
#else
#define OPENCL_RUNTIME_EXPORT
#endif // _WIN32

#define ErrCodeStr(x) case x: return #x;

const char *cl_err2str(cl_int errcode)
{
  switch (errcode)
  {
    ErrCodeStr(CL_SUCCESS                        )
    ErrCodeStr(CL_DEVICE_NOT_FOUND               )
    ErrCodeStr(CL_DEVICE_NOT_AVAILABLE           )
    ErrCodeStr(CL_COMPILER_NOT_AVAILABLE         )
    ErrCodeStr(CL_MEM_OBJECT_ALLOCATION_FAILURE  )
    ErrCodeStr(CL_OUT_OF_RESOURCES               )
    ErrCodeStr(CL_OUT_OF_HOST_MEMORY             )
    ErrCodeStr(CL_PROFILING_INFO_NOT_AVAILABLE   )
    ErrCodeStr(CL_MEM_COPY_OVERLAP               )
    ErrCodeStr(CL_IMAGE_FORMAT_MISMATCH          )
    ErrCodeStr(CL_IMAGE_FORMAT_NOT_SUPPORTED     )
    ErrCodeStr(CL_BUILD_PROGRAM_FAILURE          )
    ErrCodeStr(CL_MAP_FAILURE                    )
    ErrCodeStr(CL_MISALIGNED_SUB_BUFFER_OFFSET   )
    ErrCodeStr(CL_COMPILE_PROGRAM_FAILURE        )
    ErrCodeStr(CL_LINKER_NOT_AVAILABLE           )
    ErrCodeStr(CL_LINK_PROGRAM_FAILURE           )
    ErrCodeStr(CL_DEVICE_PARTITION_FAILED        )
    ErrCodeStr(CL_KERNEL_ARG_INFO_NOT_AVAILABLE  )
    ErrCodeStr(CL_INVALID_VALUE                  )
    ErrCodeStr(CL_INVALID_DEVICE_TYPE            )
    ErrCodeStr(CL_INVALID_PLATFORM               )
    ErrCodeStr(CL_INVALID_DEVICE                 )
    ErrCodeStr(CL_INVALID_CONTEXT                )
    ErrCodeStr(CL_INVALID_QUEUE_PROPERTIES       )
    ErrCodeStr(CL_INVALID_COMMAND_QUEUE          )
    ErrCodeStr(CL_INVALID_HOST_PTR               )
    ErrCodeStr(CL_INVALID_MEM_OBJECT             )
    ErrCodeStr(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
    ErrCodeStr(CL_INVALID_IMAGE_SIZE             )
    ErrCodeStr(CL_INVALID_SAMPLER                )
    ErrCodeStr(CL_INVALID_BINARY                 )
    ErrCodeStr(CL_INVALID_BUILD_OPTIONS          )
    ErrCodeStr(CL_INVALID_PROGRAM                )
    ErrCodeStr(CL_INVALID_PROGRAM_EXECUTABLE     )
    ErrCodeStr(CL_INVALID_KERNEL_NAME            )
    ErrCodeStr(CL_INVALID_KERNEL_DEFINITION      )
    ErrCodeStr(CL_INVALID_KERNEL                 )
    ErrCodeStr(CL_INVALID_ARG_INDEX              )
    ErrCodeStr(CL_INVALID_ARG_VALUE              )
    ErrCodeStr(CL_INVALID_ARG_SIZE               )
    ErrCodeStr(CL_INVALID_KERNEL_ARGS            )
    ErrCodeStr(CL_INVALID_WORK_DIMENSION         )
    ErrCodeStr(CL_INVALID_WORK_GROUP_SIZE        )
    ErrCodeStr(CL_INVALID_WORK_ITEM_SIZE         )
    ErrCodeStr(CL_INVALID_GLOBAL_OFFSET          )
    ErrCodeStr(CL_INVALID_EVENT_WAIT_LIST        )
    ErrCodeStr(CL_INVALID_EVENT                  )
    ErrCodeStr(CL_INVALID_OPERATION              )
    ErrCodeStr(CL_INVALID_GL_OBJECT              )
    ErrCodeStr(CL_INVALID_BUFFER_SIZE            )
    ErrCodeStr(CL_INVALID_MIP_LEVEL              )
    ErrCodeStr(CL_INVALID_GLOBAL_WORK_SIZE       )
    ErrCodeStr(CL_INVALID_PROPERTY               )
    ErrCodeStr(CL_INVALID_IMAGE_DESCRIPTOR       )
    ErrCodeStr(CL_INVALID_COMPILER_OPTIONS       )
    ErrCodeStr(CL_INVALID_LINKER_OPTIONS         )
    ErrCodeStr(CL_INVALID_DEVICE_PARTITION_COUNT )
    ErrCodeStr(CL_INVALID_PIPE_SIZE              )
    ErrCodeStr(CL_INVALID_DEVICE_QUEUE           )
    ErrCodeStr(CL_INVALID_SPEC_ID                )
    ErrCodeStr(CL_MAX_SIZE_RESTRICTION_EXCEEDED  )
    default: return "Unknown OpenCL error code!!!";
  }
}

