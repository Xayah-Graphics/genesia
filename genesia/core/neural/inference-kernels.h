#ifndef GENESIA_NEURAL_INFERENCE_KERNELS_H
#define GENESIA_NEURAL_INFERENCE_KERNELS_H

#include <cstddef>
#include <cstdint>
#include <genesia/cuda_stream.h>

namespace genesia::neural::kernels {
    void convert(const ::cuda::stream_ref stream, void* output, const void* input, std::size_t count, int source_type, int destination_type);
    void convert_layout(::cuda::stream_ref stream, void* output, const void* input, int batch, int spatial, int channels, int source, int destination);
    void activation(const ::cuda::stream_ref stream, void* output, const void* input, std::size_t count, int activation);
    void geglu(::cuda::stream_ref stream, void* output, const void* input, const void* bias, int rows, int width);
    void layer_norm(const ::cuda::stream_ref stream, void* output, const void* input, const void* weight, const void* bias, int rows, int width, float epsilon);
    void residual_norm(const ::cuda::stream_ref stream, void* output, void* residual, const void* input, const void* weight, const void* bias, int rows, int width, float epsilon);
    void group_moments(::cuda::stream_ref stream, float* statistics, const void* input, const void* second, void* joined, const void* time, const int* step, int batch, int spatial, int width, int first_width, int scalar);
    void group_norm(const ::cuda::stream_ref stream, void* output, const void* input, const void* weight, const void* bias, float* statistics, int batch, int spatial, int width, float epsilon, int scalar_type, bool silu, bool prepared = false, const void* time = nullptr, const int* step = nullptr);
    void resize(const ::cuda::stream_ref stream, void* output, const void* input, int batch, int height, int width, int channels, int scalar_type);
    void clip_attention(::cuda::stream_ref stream, void* output, const void* query, const void* key, const void* value, int batch, int queries, int keys, int heads, int query_stride, int key_stride, bool causal, const std::int32_t* query_positions);
    void attention_softmax(::cuda::stream_ref stream, void* output, const float* input, int queries, int keys);
} // namespace genesia::neural::kernels

#endif
