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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ATS_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ATS_ALLOCATOR_H_

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// An allocator for CUDA unified memory. Memory allocated with this allocator
// can be accessed from both host and device. CUDA transparently migrates dirty
// pages, which can be slow. Therefore, this allocator is intended for
// convenience in functional tests only.
class GPUATSAllocator : public Allocator {
 public:
  explicit GPUATSAllocator(Allocator* allocator,
			   PlatformGpuId platform_gpu_id, size_t threshold);
  ~GPUATSAllocator() override;
  string Name() override { return "GPUATSAllocator"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  absl::optional<AllocatorStats> GetStats() override;
  void ClearStats() override;

 private:
  Allocator* base_allocator_ = nullptr;  // owned

  se::StreamExecutor* stream_exec_;  // Not owned.

  size_t threshold_ = 1UL * 1024 * 1024 * 1024;

  mutable mutex lock_;

  AllocatorStats stats_ GUARDED_BY(lock_);

  TF_DISALLOW_COPY_AND_ASSIGN(GPUATSAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ATS_ALLOCATOR_H_
