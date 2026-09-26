#ifndef TOOLS_COMPUTE_INFERENCE_KERNELS_H
#define TOOLS_COMPUTE_INFERENCE_KERNELS_H

#include <cstddef>
#include <genesia/cuda_stream.h>

namespace tools::compute::kernels {
    void convert(::cuda::stream_ref stream, void* output, const void* input, std::size_t count, int source_type);
    void convert_layout(::cuda::stream_ref stream, void* output, const void* input, int batch, int spatial, int channels, int source_type);
    void gelu(::cuda::stream_ref stream, void* output, const void* input, std::size_t count);
    void layer_norm(::cuda::stream_ref stream, void* output, const void* input, const void* weight, const void* bias, int rows, int width, float epsilon);
} // namespace tools::compute::kernels

#endif
