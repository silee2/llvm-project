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
  switch (err)
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

// Return first GPU device
static cl_device_id getDefaultDevice() {
  static sycl::device syclDevice;
  static bool isDeviceInitialised = false;
  if (!isDeviceInitialised) {
    auto platformList = sycl::platform::get_platforms();
    for (const auto &platform : platformList) {
      auto platformName = platform.get_info<sycl::info::platform::name>();
      bool isLevelZero = platformName.find("Level-Zero") != std::string::npos;
      if (!isLevelZero)
        continue;

      syclDevice = platform.get_devices()[0];
      isDeviceInitialised = true;
      return syclDevice;
    }
    throw std::runtime_error("getDefaultDevice failed");
  } else
    return syclDevice;
}

// Return context associated we default device
static cl_context getDefaultContext() {
    cl_context clCreateContext (
const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void (CL_CALLBACK*pfn_notify)
 (const char *errinfo, const void *private_info, size_t cb, void *user_data),
void *user_data, cl_int *errcode_ret)

  static sycl::context syclContext{getDefaultDevice()};
  return syclContext;
}

struct QUEUE {
  sycl::queue syclQueue_;

  QUEUE() { syclQueue_ = sycl::queue(getDefaultContext(), getDefaultDevice()); }
};

static void *allocDeviceMemory(QUEUE *queue, size_t size, bool isShared) {
  void *memPtr = nullptr;
  if (isShared) {
    memPtr = sycl::aligned_alloc_shared(64, size, getDefaultDevice(),
                                        getDefaultContext());
  } else {
    memPtr = sycl::aligned_alloc_device(64, size, getDefaultDevice(),
                                        getDefaultContext());
  }
  if (memPtr == nullptr) {
    throw std::runtime_error("mem allocation failed!");
  }
  return memPtr;
}

static void deallocDeviceMemory(QUEUE *queue, void *ptr) {
  sycl::free(ptr, queue->syclQueue_);
}

static ze_module_handle_t loadModule(const void *data, size_t dataSize) {
  assert(data);
  cl_program clCreateProgramWithIL(
    cl_context context,
    data,
    datasize,
    cl_int* errcode_ret);

  ze_module_handle_t zeModule;
  ze_module_desc_t desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                           nullptr,
                           ZE_MODULE_FORMAT_IL_SPIRV,
                           dataSize,
                           (const uint8_t *)data,
                           nullptr,
                           nullptr};
  auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      getDefaultDevice());
  auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      getDefaultContext());
  L0_SAFE_CALL(zeModuleCreate(zeContext, zeDevice, &desc, &zeModule, nullptr));
  return zeModule;
}

static sycl::kernel *getKernel(ze_module_handle_t zeModule, const char *name) {
  assert(zeModule);
  assert(name);
  ze_kernel_handle_t zeKernel;
  sycl::kernel *syclKernel;
  ze_kernel_desc_t desc = {};
  desc.pKernelName = name;

  L0_SAFE_CALL(zeKernelCreate(zeModule, &desc, &zeKernel));
  sycl::kernel_bundle<sycl::bundle_state::executable> kernelBundle =
      sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                               sycl::bundle_state::executable>(
          {zeModule}, getDefaultContext());

  auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {kernelBundle, zeKernel}, getDefaultContext());
  syclKernel = new sycl::kernel(kernel);
  return syclKernel;
}

static void launchKernel(QUEUE *queue, sycl::kernel *kernel, size_t gridX,
                         size_t gridY, size_t gridZ, size_t blockX,
                         size_t blockY, size_t blockZ, size_t sharedMemBytes,
                         void **params, size_t paramsCount) {
  auto syclGlobalRange =
      sycl::range<3>(blockZ * gridZ, blockY * gridY, blockX * gridX);
  auto syclLocalRange = sycl::range<3>(blockZ, blockY, blockX);
  sycl::nd_range<3> syclNdRange(syclGlobalRange, syclLocalRange);

  queue->syclQueue_.submit([&](sycl::handler &cgh) {
    for (size_t i = 0; i < paramsCount; i++) {
      cgh.set_arg(static_cast<uint32_t>(i), *(static_cast<void **>(params[i])));
    }
    cgh.parallel_for(syclNdRange, *kernel);
  });
  cl_int clEnqueueNDRangeKernel (
cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
}

// Wrappers

extern "C" OPENCL_RUNTIME_EXPORT QUEUE *mgpuStreamCreate() {

  return catchAll([&]() { return new QUEUE(); });
}

extern "C" OPENCL_RUNTIME_EXPORT void mgpuStreamDestroy(QUEUE *queue) {
  catchAll([&]() { delete queue; });
}

extern "C" OPENCL_RUNTIME_EXPORT void *mgpuMemAlloc(uint64_t size, QUEUE *queue,
                                                  bool isShared) {
  return catchAll([&]() {
    return allocDeviceMemory(queue, static_cast<size_t>(size), true);
  });
}

extern "C" OPENCL_RUNTIME_EXPORT void mgpuMemFree(void *ptr, QUEUE *queue) {
  catchAll([&]() {
    if (ptr) {
      deallocDeviceMemory(queue, ptr);
    }
  });
}

extern "C" OPENCL_RUNTIME_EXPORT ze_module_handle_t
mgpuModuleLoad(const void *data, size_t gpuBlobSize) {
  return catchAll([&]() { return loadModule(data, gpuBlobSize); });
}

extern "C" OPENCL_RUNTIME_EXPORT sycl::kernel *
mgpuModuleGetFunction(ze_module_handle_t module, const char *name) {
  return catchAll([&]() { return getKernel(module, name); });
}

extern "C" OPENCL_RUNTIME_EXPORT void
mgpuLaunchKernel(sycl::kernel *kernel, size_t gridX, size_t gridY, size_t gridZ,
                 size_t blockX, size_t blockY, size_t blockZ,
                 size_t sharedMemBytes, QUEUE *queue, void **params,
                 void **extra, size_t paramsCount) {
  return catchAll([&]() {
    launchKernel(queue, kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
                 sharedMemBytes, params, paramsCount);
  });
}

extern "C" OPENCL_RUNTIME_EXPORT void mgpuStreamSynchronize(QUEUE *queue) {

  catchAll([&]() { queue->syclQueue_.wait(); });
}

extern "C" OPENCL_RUNTIME_EXPORT void
mgpuModuleUnload(ze_module_handle_t module) {

  catchAll([&]() { L0_SAFE_CALL(zeModuleDestroy(module)); });
}
