#ifndef CLASSIFIER_KERNEL_COMMON_CUH
#define CLASSIFIER_KERNEL_COMMON_CUH
#include "kernels.h"
#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
namespace classifier {
    __device__ inline float read(Tensor t, std::size_t i) {
        return t.fp32 ? static_cast<float*>(t.data)[i] : __bfloat162float(static_cast<__nv_bfloat16*>(t.data)[i]);
    }
    __device__ inline void write(Tensor t, std::size_t i, float x) {
        if (t.fp32) static_cast<float*>(t.data)[i] = x;
        else static_cast<__nv_bfloat16*>(t.data)[i] = __float2bfloat16_rn(x);
    }
    __device__ inline float rounded(float x) {
        return __bfloat162float(__float2bfloat16_rn(x));
    }
    __device__ inline float bf(const void* p, std::size_t i) {
        return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
    }
    // Philox4x32-10. A sequence is one replay; tags separate every stochastic operator.
    __device__ inline float uniform(RandomState* state, std::uint64_t index, unsigned tag) {
        uint4 c     = make_uint4(unsigned(index), unsigned(index >> 32), unsigned(state->sequence), unsigned(state->sequence >> 32) ^ tag);
        unsigned k0 = unsigned(state->seed), k1 = unsigned(state->seed >> 32);
#pragma unroll
        for (int r = 0; r < 10; ++r) {
            unsigned hi0 = __umulhi(0xD2511F53u, c.x), hi1 = __umulhi(0xCD9E8D57u, c.z);
            c = make_uint4(hi1 ^ c.y ^ k0, 0xCD9E8D57u * c.z, hi0 ^ c.w ^ k1, 0xD2511F53u * c.x);
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        return (float(c.x >> 8) + .5f) * 0x1p-24f;
    }
} // namespace classifier
#endif
