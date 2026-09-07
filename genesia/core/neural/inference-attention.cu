#include "inference-kernels.h"
#include <cuda/launch>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <genesia/cuda.h>
#include <math_constants.h>

namespace genesia::neural::kernels {
    __global__ void clip_attention_kernel(__half* output, const __half* query, const __half* key, const __half* value, const int queries, const int keys, const int heads, const int query_stride, const int key_stride, const bool causal, const std::int32_t* positions) {
        __shared__ float scores[4][96];
        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        const int row  = blockIdx.x * 4 + warp;
        if (row >= queries) return;
        const int head  = blockIdx.y % heads;
        const int batch = blockIdx.y / heads;
        const int last  = causal ? (positions ? positions[batch] : row) : keys - 1;
        float maximum   = -CUDART_INF_F;
        for (int k = lane; k < keys; k += 32) {
            float sum = 0.0F;
            for (int d = 0; d < 64; ++d) sum = fmaf(float(query[(batch * queries + row) * query_stride + head * 64 + d]), float(key[(batch * keys + k) * key_stride + head * 64 + d]), sum);
            const float score = k <= last ? sum * 0.125F : -CUDART_INF_F;
            scores[warp][k]   = score;
            maximum           = fmaxf(maximum, score);
        }
        for (int offset = 16; offset; offset >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        float denominator = 0.0F;
        for (int k = lane; k < keys; k += 32) {
            const float p   = expf(scores[warp][k] - maximum);
            scores[warp][k] = p;
            denominator += p;
        }
        for (int offset = 16; offset; offset >>= 1) denominator += __shfl_xor_sync(0xffffffff, denominator, offset);
        __syncwarp();
        for (int d = lane; d < 64; d += 32) {
            float sum = 0.0F;
            for (int k = 0; k < keys; ++k) sum = fmaf(scores[warp][k] / denominator, float(value[(batch * keys + k) * key_stride + head * 64 + d]), sum);
            output[((batch * queries + row) * heads + head) * 64 + d] = __half(sum);
        }
    }

    __global__ void softmax_kernel(__nv_bfloat16* output, const float* input, const int width) {
        __shared__ float reduction[256];
        const std::size_t row = std::size_t(blockIdx.x) * width;
        float maximum         = -CUDART_INF_F;
        for (int i = threadIdx.x; i < width; i += 256) maximum = fmaxf(maximum, input[row + i]);
        reduction[threadIdx.x] = maximum;
        __syncthreads();
        for (int offset = 128; offset; offset /= 2) {
            if (threadIdx.x < offset) reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
            __syncthreads();
        }
        maximum = reduction[0];
        __syncthreads();
        float sum = 0.0F;
        for (int i = threadIdx.x; i < width; i += 256) sum += expf(input[row + i] - maximum);
        reduction[threadIdx.x] = sum;
        __syncthreads();
        for (int offset = 128; offset; offset /= 2) {
            if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
            __syncthreads();
        }
        const float inverse = 1.0F / reduction[0];
        for (int i = threadIdx.x; i < width; i += 256) output[row + i] = __nv_bfloat16(expf(input[row + i] - maximum) * inverse);
    }

    void clip_attention(const ::cuda::stream_ref stream, void* output, const void* query, const void* key, const void* value, const int batch, const int queries, const int keys, const int heads, const int query_stride, const int key_stride, const bool causal, const std::int32_t* positions) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(dim3((queries + 3) / 4, batch * heads)), ::cuda::block_dims(128))), clip_attention_kernel, static_cast<__half*>(output), static_cast<const __half*>(query), static_cast<const __half*>(key), static_cast<const __half*>(value), queries, keys, heads, query_stride, key_stride, causal, positions);
    }

    void attention_softmax(const ::cuda::stream_ref stream, void* output, const float* input, const int queries, const int keys) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(queries), ::cuda::block_dims(256))), softmax_kernel, static_cast<__nv_bfloat16*>(output), input, keys);
    }
} // namespace genesia::neural::kernels
