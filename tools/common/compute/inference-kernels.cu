#include "inference-kernels.h"
#include <cuda/launch>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <genesia/cuda.h>

namespace tools::compute::kernels {

    template <class Function>
    void dispatch(const int scalar, Function function) {
        if (scalar == 0) {
            function.template operator()<float>();
            return;
        }
        if (scalar == 1) function.template operator()<__half>();
        else function.template operator()<__nv_bfloat16>();
    }

    template <class Input>
    __global__ void convert_kernel(__half* output, const Input* input, const std::size_t count) {
        const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) output[i] = __half(float(input[i]));
    }

    template <class I>
    __global__ void convert_layout_kernel(__half* output, const I* input, const int spatial, const int channels, const std::size_t count) {
        const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const std::size_t nchw = (i / (spatial * channels) * channels + i % channels) * spatial + i / channels % spatial;
        output[i]              = __half(float(input[nchw]));
    }

    __global__ void gelu_kernel(__half* output, const __half* input, const std::size_t count) {
        const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const float x = float(input[i]);
        output[i]     = __half(0.5F * x * (1.0F + erff(x * 0.7071067811865475F)));
    }

    template <int Width>
    __global__ void layer_norm_kernel(__half* output, const __half* input, const __half* weight, const __half* bias, const float epsilon) {
        constexpr int width = Width;
        __shared__ float partial[8];
        float values[(Width + 255) / 256];
        float sum = 0.0F;
#pragma unroll
        for (int j = 0; j < (width + 255) / 256; ++j) {
            const int c = j * 256 + threadIdx.x;
            const int i = blockIdx.x * width + c;
            values[j]   = c < width ? float(input[i]) : 0.0F;
            sum += values[j];
        }
        for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
        __syncthreads();
        sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0F;
        for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) partial[0] = sum / float(width);
        __syncthreads();
        const float mean = partial[0];
        float squared    = 0.0F;
#pragma unroll
        for (int j = 0; j < (width + 255) / 256; ++j)
            if (j * 256 + threadIdx.x < width) squared = fmaf(values[j] - mean, values[j] - mean, squared);
        __syncthreads();
        for (int offset = 16; offset; offset >>= 1) squared += __shfl_down_sync(0xffffffff, squared, offset);
        if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = squared;
        __syncthreads();
        squared = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0F;
        for (int offset = 16; offset; offset >>= 1) squared += __shfl_down_sync(0xffffffff, squared, offset);
        if (threadIdx.x == 0) partial[0] = rsqrtf(squared / float(width) + epsilon);
        __syncthreads();
        const float inverse = partial[0];
#pragma unroll
        for (int j = 0; j < (width + 255) / 256; ++j) {
            const int c = j * 256 + threadIdx.x;
            if (c >= width) continue;
            const int i = blockIdx.x * width + c;
            output[i] = __half(fmaf((values[j] - mean) * inverse, float(weight[c]), float(bias[c])));
        }
    }

    void convert(const ::cuda::stream_ref stream, void* output, const void* input, const std::size_t count, const int source) {
        dispatch(source, [&]<class I>() { ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), convert_kernel<I>, static_cast<__half*>(output), static_cast<const I*>(input), count); });
    }
    void convert_layout(const ::cuda::stream_ref stream, void* output, const void* input, const int batch, const int spatial, const int channels, const int source) {
        const std::size_t count = std::size_t(batch) * spatial * channels;
        dispatch(source, [&]<class I>() { ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), convert_layout_kernel<I>, static_cast<__half*>(output), static_cast<const I*>(input), spatial, channels, count); });
    }

    void gelu(const ::cuda::stream_ref stream, void* output, const void* input, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), gelu_kernel, static_cast<__half*>(output), static_cast<const __half*>(input), count);
    }
    void layer_norm(const ::cuda::stream_ref stream, void* output, const void* input, const void* weight, const void* bias, const int rows, const int width, const float epsilon) {
        const auto launch = [&]<int Width>() { ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(rows), ::cuda::block_dims(256))), layer_norm_kernel<Width>, static_cast<__half*>(output), static_cast<const __half*>(input), static_cast<const __half*>(weight), static_cast<const __half*>(bias), epsilon); };
        switch (width) {
        case 192: launch.template operator()<192>(); break;
        case 384: launch.template operator()<384>(); break;
        case 768: launch.template operator()<768>(); break;
        case 1536: launch.template operator()<1536>(); break;
        case 3072: launch.template operator()<3072>(); break;
        }
    }
} // namespace tools::compute::kernels
