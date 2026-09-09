#ifndef GENESIA_GENERATION_REPAINT_KERNELS_H
#define GENESIA_GENERATION_REPAINT_KERNELS_H

#include <cstdint>
#include <genesia/cuda_stream.h>

namespace genesia::repaint_kernels {
    struct Geometry final {
        int width, height;
        int x, y, crop_width, crop_height;
        int resized_width, resized_height;
        int work_width, work_height;
    };

    void prepare(::cuda::stream_ref stream, std::uint8_t* work, std::uint8_t* latent_mask, float* alpha, float* scratch, const std::uint8_t* original, const std::uint8_t* mask, Geometry geometry, int feather);
    void composite(::cuda::stream_ref stream, std::uint8_t* output, float* scratch, const std::uint8_t* work, const std::uint8_t* original, const float* alpha, Geometry geometry);
} // namespace genesia::repaint_kernels

#endif
