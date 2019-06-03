/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef GOOGLE_CUDA
#include "cuda/include/cuda.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif  // GOOGLE_CUDA

#include <stdio.h>
#include "tensorflow/core/common_runtime/gpu/gpu_ats_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

GPUATSAllocator::GPUATSAllocator(Allocator* allocator,
				 PlatformGpuId platform_gpu_id,
                                 size_t threshold)
  : base_allocator_(allocator) {
  stream_exec_ =
    GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
  threshold_ = threshold;
  stats_.bytes_limit = 0;
  
  printf("GPUATSAllocator::GPUATSAllocator(threshold=%ld): called\n",
         threshold);
  fflush(stdout);
}

GPUATSAllocator::~GPUATSAllocator() { delete base_allocator_; }

void* GPUATSAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  void* ptr = nullptr;
  if ((threshold_ > 0) && (num_bytes < threshold_)) {
    se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
    CUdeviceptr rv = 0;
    CUresult res = cuMemAlloc(&rv, num_bytes);
    if (res != CUDA_SUCCESS) {
      LOG(ERROR) << "cuMemAlloc failed to allocate " << num_bytes;
      printf("GPUATSAllocator::AllocateRaw(alignment=%ld, num_bytes=%ld):"
           " return %p (FAIL)\n", alignment, num_bytes, nullptr);
      return nullptr;
    }
    ptr = reinterpret_cast<void*>(rv); 
    printf("GPUATSAllocator::AllocateRaw(alignment=%ld, num_bytes=%ld): return "
	   "%p GPUMEM (res=%d)\n", alignment, num_bytes, ptr, res);
    fflush(stdout);
  } else {
    int res = posix_memalign(&ptr, 512, num_bytes);
    printf("GPUATSAllocator::AllocateRaw(alignment=%ld, num_bytes=%ld): return "
	   "%p ATSMEM (res=%d)\n", alignment, num_bytes, ptr, res);
    fflush(stdout);
  }
  // Update stats.
  ++stats_.num_allocs;
  stats_.bytes_in_use += 0;
  stats_.peak_bytes_in_use =
    std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
  stats_.largest_alloc_size =
    std::max<std::size_t>(stats_.largest_alloc_size, num_bytes);
		  
  return ptr;
}

void GPUATSAllocator::DeallocateRaw(void* ptr) {
  CUresult res = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
  if (res == CUDA_ERROR_INVALID_VALUE) {
    printf("GPUATSAllocator::DeallocateRaw(ptr=%p): return ATSMEM (res=%d)\n",
           ptr, res);
    fflush(stdout);
    free(ptr);
  } else {
    printf("GPUATSAllocator::DeallocateRaw(ptr=%p): return GPUMEM (res=%d)\n",
	   ptr, res);
    fflush(stdout);
  }
  stats_.bytes_in_use -= 0;
  return;
}

absl::optional<AllocatorStats> GPUATSAllocator::GetStats() {
  mutex_lock l(lock_);
  return stats_;
}

void GPUATSAllocator::ClearStats() {
  mutex_lock l(lock_);
  stats_.num_allocs = 0;
  stats_.peak_bytes_in_use = stats_.bytes_in_use;
  stats_.largest_alloc_size = 0;
}

}  // namespace tensorflow
