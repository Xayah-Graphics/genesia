#include "kernels.h"
#include <algorithm>
#include <cub/block/block_scan.cuh>
#include <cuda/launch>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <genesia/cuda.h>

namespace genesia::sdxl::kernels {
    __global__ void embedding_kernel(__half* output, const __half* token, const __half* position, const std::int32_t* ids, const int rows, const int width) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < rows * width) output[i] = __half(float(token[ids[i / width] * width + i % width]) + float(position[i % (77 * width)]));
    }

    __global__ void gather_kernel(__half* output, const __half* input, const std::int32_t* rows, const int count, const int width) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count * width) output[i] = input[rows[i / width] * width + i % width];
    }

    __global__ void weighted_kernel(__half* output, const __half* input, const __half* empty, const float* weights, const int rows, const int width) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= rows * width) return;
        const float weight = weights[i / width];
        const float base   = float(empty[i % (77 * width)]);
        output[i]          = __half(fmaf(weight, float(input[i]) - base, base));
    }

    __global__ void conditioning_kernel(__half* output, const __half* clip_l, const __half* clip_g, const int positive, const int negative, const int length) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= 2 * length * 2048) return;
        const int batch = i / (length * 2048);
        const int token = i / 2048 % length;
        const int row   = batch * positive * 77 + token;
        const int c     = i % 2048;
        output[i]       = token < (batch ? negative : positive) * 77 ? (c < 768 ? clip_l[row * 768 + c] : clip_g[row * 1280 + c - 768]) : __half(0.0F);
    }

    template <class T>
    __global__ void time_embedding_kernel(T* output, const float* times, const int count, const int width) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count * width) return;
        const float frequency = expf(-9.210340371976184F * float(i % (width / 2)) / float(width / 2));
        const float angle     = times[i / width] * frequency;
        output[i]             = T(i % width < width / 2 ? cosf(angle) : sinf(angle));
    }

    __global__ void conditions_kernel(__half* output, const __half* pooled, const float* geometry) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < 5632) output[i] = __half(i % 2816 < 1280 ? float(pooled[i / 2816 * 1280 + i % 2816]) : geometry[i % 2816 - 1280]);
    }

    __global__ void time_condition_kernel(__half* output, const __half* times, const __half* labels, const int steps) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= steps * 2560) return;
        const float value = float(times[i / 2560 * 1280 + i % 1280]) + float(labels[i % 2560]);
        output[i]         = __half(float(value) / (1.0F + expf(-float(value))));
    }

    __global__ void training_kernel(float* training) {
        __shared__ cub::BlockScan<float, 256>::TempStorage storage;
        float values[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i      = threadIdx.x * 4 + j;
            const float beta = fmaf(float(i) / 999.0F, sqrtf(0.012F) - sqrtf(0.00085F), sqrtf(0.00085F));
            values[j]        = i < 1000 ? log1pf(-beta * beta) : 0.0F;
        }
        cub::BlockScan<float, 256>(storage).InclusiveSum(values, values);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (threadIdx.x * 4 + j < 1000) training[threadIdx.x * 4 + j] = sqrtf(expm1f(-values[j]));
    }

    __global__ void schedule_kernel(SamplingStep* output, float* times, const float* training, const int steps, const int total) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i > steps) return;
        const int position = total - steps + i;
        const int time = 999 - int(static_cast<long long>(position) * 1000 / total);
        const float sigma = i == steps ? 0.0F : training[time];
        const float next = i + 1 >= steps ? 0.0F : training[999 - int(static_cast<long long>(position + 1) * 1000 / total)];
        output[i]         = {sigma, next - sigma, rsqrtf(fmaf(sigma, sigma, 1.0F))};
        if (i < steps) times[i] = float(time);
    }

    __global__ void initialize_kernel(float* state, __half* input, int* step, const std::uint64_t* seed, const SamplingStep* schedule, const int count, const float* source, const bool full_noise) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count / 4) return;
        uint4 counter{static_cast<unsigned>(i), 0, 0, 0};
        uint2 key{static_cast<unsigned>(*seed), static_cast<unsigned>(*seed >> 32)};
#pragma unroll
        for (int round = 0; round < 10; ++round) {
            const unsigned high0 = __umulhi(0xd2511f53u, counter.x);
            const unsigned high1 = __umulhi(0xcd9e8d57u, counter.z);
            counter              = {high1 ^ counter.y ^ key.x, 0xcd9e8d57u * counter.z, high0 ^ counter.w ^ key.y, 0xd2511f53u * counter.x};
            key.x += 0x9e3779b9u;
            key.y += 0xbb67ae85u;
        }
        float sine0, cosine0, sine1, cosine1;
        sincosf(float(counter.y) * 0x1p-32F * 6.283185307179586F, &sine0, &cosine0);
        sincosf(float(counter.w) * 0x1p-32F * 6.283185307179586F, &sine1, &cosine1);
        const float radius0 = sqrtf(-2.0F * logf((float(counter.x) + 1.0F) * 0x1p-32F));
        const float radius1 = sqrtf(-2.0F * logf((float(counter.z) + 1.0F) * 0x1p-32F));
        const float scale = full_noise ? sqrtf(fmaf(schedule[0].sigma, schedule[0].sigma, 1.0F)) : schedule[0].sigma;
        float4 values{radius0 * sine0 * scale, radius0 * cosine0 * scale, radius1 * sine1 * scale, radius1 * cosine1 * scale};
        if (source) {
            const float4 original = reinterpret_cast<const float4*>(source)[i];
            values.x += original.x;
            values.y += original.y;
            values.z += original.z;
            values.w += original.w;
        }
        reinterpret_cast<float4*>(state)[i] = values;
        const float data[4]{values.x, values.y, values.z, values.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) input[i * 4 + j] = input[count + i * 4 + j] = __half(data[j] * schedule[0].inverse);
        if (i == 0) *step = 0;
    }

    __global__ void snapshot_begin_kernel(int* selected, SnapshotSlot* slots) {
        *selected = -1;
        for (int i = 0; i < 3; ++i) {
            if (::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{slots[i].state}.load(::cuda::memory_order_acquire) == std::uint32_t(SnapshotState::requested)) {
                *selected = i;
                break;
            }
        }
    }

    __global__ void snapshot_publish_kernel(const int* selected, SnapshotSlot* slots, const int* step) {
        const int i = *selected;
        if (i < 0) return;
        slots[i].step = *step + 1;
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{slots[i].state}.store(std::uint32_t(SnapshotState::ready), ::cuda::memory_order_release);
    }

    template <bool preview>
    __global__ void euler_kernel(float* state, __half* input, const __half* epsilon, const SamplingStep* schedule, const int* step, const float cfg, const int count, float* snapshots, const int* selected) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const int index        = *step;
        const float derivative = fmaf(cfg, float(epsilon[i]) - float(epsilon[i + count]), float(epsilon[i + count]));
        if constexpr (preview) {
            const int slot = *selected;
            if (slot >= 0) snapshots[slot * count + i] = fmaf(-schedule[index].sigma, derivative, state[i]);
        }
        const float value = fmaf(schedule[index].delta, derivative, state[i]);
        state[i]          = value;
        if (schedule[index + 1].sigma != 0.0F) input[i] = input[i + count] = __half(value * schedule[index + 1].inverse);
    }

    __global__ void advance_kernel(int* step, const int count, const cudaGraphConditionalHandle loop, const cudaGraphConditionalHandle decode, Control* control) {
        const int completed = ++*step;
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control->completed}.store(completed, ::cuda::memory_order_release);
        const bool stopped = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control->cancel}.load(::cuda::memory_order_acquire) != 0;
        cudaGraphSetConditional(loop, !stopped && completed < count);
        cudaGraphSetConditional(decode, !stopped && completed == count);
        if (stopped || completed == count) ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control->stage}.store(static_cast<std::uint32_t>(stopped ? Stage::cancelled : Stage::decoding), ::cuda::memory_order_release);
    }

    __global__ void image_encode_kernel(__nv_bfloat16* output, const std::uint8_t* input, const int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) output[i] = __nv_bfloat16(float(input[i]) * (2.0F / 255.0F) - 1.0F);
    }

    __global__ void encoder_pad_kernel(__nv_bfloat16* output, const __nv_bfloat16* input, const int height, const int width, const int channels) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= (height + 1) * (width + 1) * channels) return;
        const int x = i / channels % (width + 1);
        const int y = i / channels / (width + 1);
        output[i] = x < width && y < height ? input[(y * width + x) * channels + i % channels] : __nv_bfloat16(0.0F);
    }

    __global__ void latent_encode_kernel(float* output, const __nv_bfloat16* moments, const int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) output[i] = float(moments[i / 4 * 8 + i % 4]) * 0.13025F;
    }

    __global__ void latent_decode_kernel(__nv_bfloat16* output, const float* input, const int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) output[i] = __nv_bfloat16(input[i] / 0.13025F);
    }

    __global__ void pixels_kernel(std::uint8_t* output, const __nv_bfloat16* input, const int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) output[i] = static_cast<std::uint8_t>(fminf(fmaxf(float(input[i]) * 0.5F + 0.5F, 0.0F), 1.0F) * 255.0F);
    }


    void embedding(const ::cuda::stream_ref stream, void* output, const void* token, const void* position, const std::int32_t* ids, const int rows, const int width) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((rows * width + 255) / 256), ::cuda::block_dims(256))), embedding_kernel, static_cast<__half*>(output), static_cast<const __half*>(token), static_cast<const __half*>(position), ids, rows, width);
    }
    void gather_rows(const ::cuda::stream_ref stream, void* output, const void* input, const std::int32_t* rows, const int count, const int width) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count * width + 255) / 256), ::cuda::block_dims(256))), gather_kernel, static_cast<__half*>(output), static_cast<const __half*>(input), rows, count, width);
    }
    void weighted_text(const ::cuda::stream_ref stream, void* output, const void* input, const void* empty, const float* weights, const int rows, const int width) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((rows * width + 255) / 256), ::cuda::block_dims(256))), weighted_kernel, static_cast<__half*>(output), static_cast<const __half*>(input), static_cast<const __half*>(empty), weights, rows, width);
    }
    void conditioning(const ::cuda::stream_ref stream, void* output, const void* clip_l, const void* clip_g, const int positive, const int negative, const int length) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((2 * length * 2048 + 255) / 256), ::cuda::block_dims(256))), conditioning_kernel, static_cast<__half*>(output), static_cast<const __half*>(clip_l), static_cast<const __half*>(clip_g), positive, negative, length);
    }
    void time_embedding(const ::cuda::stream_ref stream, void* output, const float* times, const int count, const int width, const int scalar) {
        if (scalar == 0) ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count * width + 255) / 256), ::cuda::block_dims(256))), time_embedding_kernel<float>, static_cast<float*>(output), times, count, width);
        else ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count * width + 255) / 256), ::cuda::block_dims(256))), time_embedding_kernel<__half>, static_cast<__half*>(output), times, count, width);
    }
    void conditions(const ::cuda::stream_ref stream, void* output, const void* pooled, const void* geometry) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(22), ::cuda::block_dims(256))), conditions_kernel, static_cast<__half*>(output), static_cast<const __half*>(pooled), static_cast<const float*>(geometry));
    }
    void time_condition(const ::cuda::stream_ref stream, void* output, const void* times, const void* labels, const int steps) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((steps * 2560 + 255) / 256), ::cuda::block_dims(256))), time_condition_kernel, static_cast<__half*>(output), static_cast<const __half*>(times), static_cast<const __half*>(labels), steps);
    }
    void training_sigmas(const ::cuda::stream_ref stream, float* output) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(1), ::cuda::block_dims(256))), training_kernel, output);
    }
    void prepare_schedule(const ::cuda::stream_ref stream, SamplingStep* output, float* times, const float* training, const int steps, const float denoise) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((steps + 256) / 256), ::cuda::block_dims(256))), schedule_kernel, output, times, training, steps, static_cast<int>(steps / double(denoise)));
    }
    void initialize(const ::cuda::stream_ref stream, float* state, void* input, int* step, const std::uint64_t* seed, const SamplingStep* schedule, const int count, const float* source, const bool full_noise) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count / 4 + 255) / 256), ::cuda::block_dims(256))), initialize_kernel, state, static_cast<__half*>(input), step, seed, schedule, count, source, full_noise);
    }
    void snapshot_begin(const ::cuda::stream_ref stream, int* selected, SnapshotSlot* slots) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(1), ::cuda::block_dims(1))), snapshot_begin_kernel, selected, slots);
    }

    void snapshot_publish(const ::cuda::stream_ref stream, const int* selected, SnapshotSlot* slots, const int* step) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(1), ::cuda::block_dims(1))), snapshot_publish_kernel, selected, slots, step);
    }

    void euler(const ::cuda::stream_ref stream, float* state, void* input, const void* epsilon, const SamplingStep* schedule, const int* step, const float cfg, const int count, float* snapshots, const int* selected) {
        const auto config = ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256)));
        if (snapshots) ::cuda::launch(stream, config, euler_kernel<true>, state, static_cast<__half*>(input), static_cast<const __half*>(epsilon), schedule, step, cfg, count, snapshots, selected);
        else ::cuda::launch(stream, config, euler_kernel<false>, state, static_cast<__half*>(input), static_cast<const __half*>(epsilon), schedule, step, cfg, count, snapshots, selected);
    }
    void advance(const ::cuda::stream_ref stream, int* step, const int count, const cudaGraphConditionalHandle loop, const cudaGraphConditionalHandle decode, Control* control) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(1), ::cuda::block_dims(1))), advance_kernel, step, count, loop, decode, control);
    }
    void image_encode(const ::cuda::stream_ref stream, void* output, const std::uint8_t* pixels, const int count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), image_encode_kernel, static_cast<__nv_bfloat16*>(output), pixels, count);
    }
    void encoder_pad(const ::cuda::stream_ref stream, void* output, const void* input, const int height, const int width, const int channels) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(((height + 1) * (width + 1) * channels + 255) / 256), ::cuda::block_dims(256))), encoder_pad_kernel, static_cast<__nv_bfloat16*>(output), static_cast<const __nv_bfloat16*>(input), height, width, channels);
    }
    void latent_encode(const ::cuda::stream_ref stream, float* output, const void* moments, const int count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), latent_encode_kernel, output, static_cast<const __nv_bfloat16*>(moments), count);
    }
    void latent_decode(const ::cuda::stream_ref stream, void* output, const float* latent, const int count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), latent_decode_kernel, static_cast<__nv_bfloat16*>(output), latent, count);
    }
    void pixels(const ::cuda::stream_ref stream, std::uint8_t* output, const void* input, const int count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), pixels_kernel, output, static_cast<const __nv_bfloat16*>(input), count);
    }
} // namespace genesia::sdxl::kernels
