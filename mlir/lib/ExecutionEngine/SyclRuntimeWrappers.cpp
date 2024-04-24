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

} // namespace

thread_local static int32_t defaultDevice = 0;

static sycl::device getDefaultDevice() {
  auto platformList = sycl::platform::get_platforms();
  for (const auto &platform : platformList) {
    if(platform.get_backend() == sycl::backend::opencl) {
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

static cl_program loadModule(const void *data, size_t dataSize) {
  assert(data);
  cl_int err = CL_SUCCESS;
  auto oclDev = sycl::get_native<sycl::backend::opencl>(
      getDefaultDevice());
  auto oclCtx = sycl::get_native<sycl::backend::opencl>(
      getDefaultContext());
  cl_program oclProgram = clCreateProgramWithIL(oclCtx, data, dataSize, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to load module");
  }
  clBuildProgram(oclProgram, 1, &oclDev, nullptr, nullptr, nullptr);
  return oclProgram;
}

static sycl::kernel *getKernel(cl_program oclProgram, const char *name) {
  assert(oclProgram);
  assert(name);
  cl_int err = CL_SUCCESS;
  cl_kernel oclKernel = clCreateKernel(oclProgram, name, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create kernel");
  }
  return new sycl::kernel(sycl::make_kernel<sycl::backend::opencl>(oclKernel, getDefaultContext()));
}

static void launchKernel(sycl::queue *queue, sycl::kernel *kernel, size_t gridX,
                         size_t gridY, size_t gridZ, size_t blockX,
                         size_t blockY, size_t blockZ, size_t sharedMemBytes,
                         void **params, size_t paramsCount) {
  auto syclGlobalRange =
      sycl::range<3>(blockZ * gridZ, blockY * gridY, blockX * gridX);
  auto syclLocalRange = sycl::range<3>(blockZ, blockY, blockX);
  sycl::nd_range<3> syclNdRange(syclGlobalRange, syclLocalRange);
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
    cgh.parallel_for(syclNdRange, *kernel);
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

extern "C" SYCL_RUNTIME_EXPORT cl_program
mgpuModuleLoad(const void *data, size_t gpuBlobSize) {
  return catchAll([&]() { return loadModule(data, gpuBlobSize); });
}

extern "C" SYCL_RUNTIME_EXPORT sycl::kernel *
mgpuModuleGetFunction(cl_program module, const char *name) {
  return catchAll([&]() { return getKernel(module, name); });
}

extern "C" SYCL_RUNTIME_EXPORT void
mgpuLaunchKernel(sycl::kernel *kernel, size_t gridX, size_t gridY, size_t gridZ,
                 size_t blockX, size_t blockY, size_t blockZ,
                 size_t sharedMemBytes, sycl::queue *queue, void **params,
                 void ** /*extra*/, size_t paramsCount) {
  return catchAll([&]() {
    launchKernel(queue, kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
                 sharedMemBytes, params, paramsCount);
  });
}

extern "C" SYCL_RUNTIME_EXPORT void mgpuStreamSynchronize(sycl::queue *queue) {

  catchAll([&]() { queue->wait(); });
}

extern "C" SYCL_RUNTIME_EXPORT void
mgpuModuleUnload(cl_program oclProgram) {

  catchAll([&]() { clReleaseProgram(oclProgram); });
}
