// precision.cuh — bf16/f32 mixed-precision support header
//
// Include this in every fused kernel .cu file. Compile normally for f32,
// or with -DUSE_BF16 for bf16 I/O with f32 internal accumulators.
//
// bf16 halves global memory bandwidth (the bottleneck for sequential
// scan kernels) while f32 accumulators maintain numerical stability.
//
// Usage in kernel code:
//   - I/O pointer types: `const io_type*`, `io_type*`
//   - Global reads:  `float val = IO_LOAD(ptr, idx)`
//   - Global writes: `IO_STORE(ptr, idx, float_val)`
//   - All internal computation stays `float`
//   - Shared memory stays `float` (bf16 benefit is from halved global bandwidth)

#pragma once
#include <cuda_runtime.h>

#ifdef USE_BF16
  #include <cuda_bf16.h>
  typedef __nv_bfloat16 io_type;
  #define IO_LOAD(ptr, i)      __bfloat162float((ptr)[(i)])
  #define IO_STORE(ptr, i, v)  (ptr)[(i)] = __float2bfloat16(v)
  #define IO_SIZEOF            2
  #define PRECISION_SUFFIX     "bf16"
#else
  typedef float io_type;
  #define IO_LOAD(ptr, i)      (ptr)[(i)]
  #define IO_STORE(ptr, i, v)  (ptr)[(i)] = (v)
  #define IO_SIZEOF            4
  #define PRECISION_SUFFIX     "f32"
#endif

// FFI buffer type alias — use in EXLA FFI sections
#ifdef EXLA_FFI
  #ifdef USE_BF16
    #define FFI_IO_TYPE ffi::BF16
  #else
    #define FFI_IO_TYPE ffi::F32
  #endif
#endif
