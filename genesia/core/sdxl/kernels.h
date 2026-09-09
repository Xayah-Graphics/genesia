#ifndef GENESIA_GENERATIVE_SDXL_KERNELS_H
#define GENESIA_GENERATIVE_SDXL_KERNELS_H

#include "control.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <genesia/cuda_stream.h>

namespace genesia::sdxl::kernels {
    struct SamplingStep {
        float sigma;
        float delta;
        float inverse;
    };
    void embedding(const ::cuda::stream_ref stream, void* output, const void* token, const void* position, const std::int32_t* ids, const int rows, const int width);
    void gather_rows(const ::cuda::stream_ref stream, void* output, const void* input, const std::int32_t* rows, const int count, const int width);
    void weighted_text(const ::cuda::stream_ref stream, void* output, const void* input, const void* empty, const float* weights, const int rows, const int width);
    void conditioning(const ::cuda::stream_ref stream, void* output, const void* clip_l, const void* clip_g, const int positive, const int negative, const int length);
    void time_embedding(const ::cuda::stream_ref stream, void* output, const float* times, const int count, const int width, const int scalar);
    void conditions(const ::cuda::stream_ref stream, void* output, const void* pooled, const void* geometry);
    void time_condition(const ::cuda::stream_ref stream, void* output, const void* times, const void* labels, const int steps);
    void training_sigmas(const ::cuda::stream_ref stream, float* output);
    void prepare_schedule(::cuda::stream_ref stream, SamplingStep* output, float* times, const float* training, int steps, float denoise = 1);
    void initialize(::cuda::stream_ref stream, float* state, void* input, int* step, const std::uint64_t* seed, const SamplingStep* schedule, int count, const float* source = nullptr, bool full_noise = true);
    void snapshot_begin(::cuda::stream_ref stream, int* selected, SnapshotSlot* slots);
    void snapshot_publish(::cuda::stream_ref stream, const int* selected, SnapshotSlot* slots, const int* step);
    void euler(const ::cuda::stream_ref stream, float* state, void* input, const void* epsilon, const SamplingStep* schedule, const int* step, const float cfg, const int count, float* snapshots = nullptr, const int* selected = nullptr);
    void advance(::cuda::stream_ref stream, int* step, int count, cudaGraphConditionalHandle loop, cudaGraphConditionalHandle decode, Control* control);
    void image_encode(::cuda::stream_ref stream, void* output, const std::uint8_t* pixels, int count);
    void encoder_pad(::cuda::stream_ref stream, void* output, const void* input, int height, int width, int channels);
    void latent_encode(::cuda::stream_ref stream, float* output, const void* moments, int count);
    void latent_decode(const ::cuda::stream_ref stream, void* output, const float* latent, const int count);
    void pixels(const ::cuda::stream_ref stream, std::uint8_t* output, const void* input, const int count);
} // namespace genesia::sdxl::kernels

#endif
