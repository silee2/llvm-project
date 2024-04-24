//===- SyclRuntimeWrappers.cpp - MLIR SYCL wrapper library ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements wrappers around the sycl runtime library with C linkage
//
//===----------------------------------------------------------------------===//

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <type_traits>

#ifdef _WIN32
#define SYCL_RUNTIME_EXPORT __declspec(dllexport)
#else
#define SYCL_RUNTIME_EXPORT
#endif // _WIN32

namespace {

template <typename F>
auto catchAll(F &&func) {
  try {
    return func();
  } catch (const std::exception &e) {
    fprintf(stdout, "An exception was thrown: %s\n", e.what());
    fflush(stdout);
    abort();
  } catch (...) {
    fprintf(stdout, "An unknown exception was thrown\n");
    fflush(stdout);
    abort();
  }
}

[[noreturn]] static void reportError(std::string &&str) {
  throw std::runtime_error(std::move(str));
}

static std::string getClErrorString(cl_int val) {
#define CL_ENUM_VAL(arg)                                                       \
  case arg:                                                                    \
    return #arg
  switch (val) {
    CL_ENUM_VAL(CL_SUCCESS);
    CL_ENUM_VAL(CL_BUILD_PROGRAM_FAILURE);
    CL_ENUM_VAL(CL_INVALID_CONTEXT);
    CL_ENUM_VAL(CL_INVALID_DEVICE);
    CL_ENUM_VAL(CL_INVALID_VALUE);
    CL_ENUM_VAL(CL_OUT_OF_RESOURCES);
    CL_ENUM_VAL(CL_OUT_OF_HOST_MEMORY);
    CL_ENUM_VAL(CL_INVALID_OPERATION);
    CL_ENUM_VAL(CL_INVALID_BINARY);
  default:
    return "Unknown error: " + std::to_string(val);
  }
#undef CL_ENUM_VAL
}

static void checkClResult(const char *func, cl_int res) {
  if (res != CL_SUCCESS)
    reportError(std::string(func) + " failed: " + getClErrorString(res));
}

#define CHECK_CL_RESULT(arg) checkClResult(#arg, arg)

struct ClProgramDeleter {
  void operator()(cl_program program) const {
    catchAll(
        [&]() { CHECK_CL_RESULT(clReleaseProgram(program)); });
  }
};
using ClProgram =
    std::unique_ptr<std::remove_pointer_t<cl_program>, ClProgramDeleter>;

struct ClKernelDeleter {
  void operator()(cl_kernel kernel) const {
    catchAll([&]() { CHECK_CL_RESULT(clReleaseKernel(kernel)); });
  }
};
using ClKernel =
    std::unique_ptr<std::remove_pointer_t<cl_kernel>, ClKernelDeleter>;

static constexpr const auto ze_be = sycl::backend::ext_oneapi_level_zero;
static constexpr const auto cl_be = sycl::backend::opencl;

} // namespace

struct GPUModule {
  sycl::queue *queue = nullptr;
  sycl::kernel_bundle<sycl::bundle_state::executable> kernelBundle;
};

struct GPUKernel {
  sycl::queue *queue = nullptr;
  sycl::kernel syclKernel;
  uint32_t maxWgSize = 0;
};

thread_local static int32_t defaultDevice = 0;

static sycl::device getDefaultDevice() {
  auto platformList = sycl::platform::get_platforms();
  for (const auto &platform : platformList) {
    if(platform.get_backend() == cl_be) {
      auto gpuDevices = platform.get_devices(sycl::info::device_type::gpu);
      if(gpuDevices.size() > defaultDevice) {
        return gpuDevices[defaultDevice];
      } else if(gpuDevices.size() > 0) {
        throw std::runtime_error("getDefaultDevice failed: Device id exceeds GPU platform gpu count!");
      }
    }
  }
  throw std::runtime_error("getDefaultDevice failed: No GPU platform found!");
}

static sycl::context getDefaultContext() {
  static sycl::context syclContext{getDefaultDevice()};
  return syclContext;
}

static void *allocDeviceMemory(sycl::queue *queue, size_t size, bool isShared) {
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

static void deallocDeviceMemory(sycl::queue *queue, void *ptr) {
  if (queue == nullptr) {
    queue = new sycl::queue(getDefaultContext(), getDefaultDevice());
  }
  sycl::free(ptr, *queue);
}

static GPUModule* loadModule(const void *data, size_t dataSize) {
  assert(data);
  cl_int err = CL_SUCCESS;
  auto oclDev = sycl::get_native<cl_be>(
      getDefaultDevice());
  auto oclCtx = sycl::get_native<cl_be>(
      getDefaultContext());
  ClProgram program(clCreateProgramWithIL(oclCtx, data, dataSize, &err));
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to load module");
  }
  clBuildProgram(program.get(), 1, &oclDev, "", nullptr, nullptr);
  auto kernelBundle =
      sycl::make_kernel_bundle<cl_be, sycl::bundle_state::executable>(
          program.release(), getDefaultContext());
  return new GPUModule{new sycl::queue(getDefaultContext(), getDefaultDevice()), std::move(kernelBundle)};
}

static void destroyModule(GPUModule *mod) { delete mod; }

static GPUKernel *getKernel(GPUModule* mod, const char *name) {
  assert(mod);
  assert(name);
  auto queue = mod->queue;
  auto ctx = queue->get_context();
  auto maxWgSize = static_cast<uint32_t>(
      queue->get_device().get_info<sycl::info::device::max_work_group_size>());
  auto clProgram = sycl::get_native<cl_be>(mod->kernelBundle).front();
  cl_int err = CL_SUCCESS;
  ClKernel clKernel(clCreateKernel(clProgram, name, &err));
  checkClResult("clCreateKernel", err);
  auto syclKernel = sycl::make_kernel<cl_be>(clKernel.release(), ctx);
  return new GPUKernel{queue, std::move(syclKernel), maxWgSize};
}

static void destroyKernel(GPUKernel *kernel) { delete kernel; }

static sycl::kernel getSYCLKernel(GPUKernel *kernel) { return kernel->syclKernel; }

static void launchKernel(GPUKernel *kernel, size_t gridX,
                         size_t gridY, size_t gridZ, size_t blockX,
                         size_t blockY, size_t blockZ, size_t sharedMemBytes,
                         void **params, size_t paramsCount) {
  assert(kernel);
  auto syclGlobalRange =
      sycl::range<3>(blockZ * gridZ, blockY * gridY, blockX * gridX);
  auto syclLocalRange = sycl::range<3>(blockZ, blockY, blockX);
  sycl::nd_range<3> syclNdRange(syclGlobalRange, syclLocalRange);
  auto syclKernel = getSYCLKernel(kernel);
  auto queue = kernel->queue;
#if 0
  fprintf(stdout, "gridX: %d, ", gridX);
  fprintf(stdout, "gridY: %d, ", gridY);
  fprintf(stdout, "gridZ: %d, ", gridZ);
  fprintf(stdout, "blockX: %d, ", blockX);
  fprintf(stdout, "blockY: %d, ", blockY);
  fprintf(stdout, "blockZ: %d, ", blockZ);
  fprintf(stdout, "paramsCount: %d\n", paramsCount);
  fflush(stdout);
#endif
  queue->submit([&](sycl::handler &cgh) {
    for (size_t i = 0; i < paramsCount; i++) {
      cgh.set_arg(static_cast<uint32_t>(i), *(static_cast<void **>(params[i])));
    }
    fprintf(stdout, "After set_arg\n");
    fflush(stdout);
    cgh.parallel_for(syclNdRange, syclKernel);
  });
}

// Wrappers

extern "C" SYCL_RUNTIME_EXPORT sycl::queue *mgpuStreamCreate() {

  return catchAll([&]() {
    sycl::queue *queue =
        new sycl::queue(getDefaultContext(), getDefaultDevice());
    return queue;
  });
}

extern "C" SYCL_RUNTIME_EXPORT void mgpuStreamDestroy(sycl::queue *queue) {
  catchAll([&]() { delete queue; });
}

extern "C" SYCL_RUNTIME_EXPORT void *
mgpuMemAlloc(uint64_t size, sycl::queue *queue, bool isShared) {
  return catchAll([&]() {
    return allocDeviceMemory(queue, static_cast<size_t>(size), true);
  });
}

extern "C" SYCL_RUNTIME_EXPORT void mgpuMemFree(void *ptr, sycl::queue *queue) {
  catchAll([&]() {
    if (ptr) {
      deallocDeviceMemory(queue, ptr);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT GPUModule *
mgpuModuleLoad(const void *data, size_t gpuBlobSize) {
  return catchAll([&]() { return loadModule(data, gpuBlobSize); });
}

extern "C" SYCL_RUNTIME_EXPORT GPUKernel *
mgpuModuleGetFunction(GPUModule *module, const char *name) {
  return catchAll([&]() { return getKernel(module, name); });
}

extern "C" SYCL_RUNTIME_EXPORT void
mgpuLaunchKernel(GPUKernel *kernel, size_t gridX, size_t gridY, size_t gridZ,
                 size_t blockX, size_t blockY, size_t blockZ,
                 size_t sharedMemBytes, sycl::queue *queue, void **params,
                 void ** /*extra*/, size_t paramsCount) {
  return catchAll([&]() {
    launchKernel(kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
                 sharedMemBytes, params, paramsCount);
  });
}

extern "C" SYCL_RUNTIME_EXPORT void mgpuStreamSynchronize(sycl::queue *queue) {

  catchAll([&]() { queue->wait(); });
}

extern "C" SYCL_RUNTIME_EXPORT void
mgpuModuleUnload(GPUModule *module) {

  catchAll([&]() { destroyModule(module); });
}
