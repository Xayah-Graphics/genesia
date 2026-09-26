#ifndef FOREGROUND_BIREFNET_KERNELS_H
#define FOREGROUND_BIREFNET_KERNELS_H
#include <compute/tensor.h>
#include <genesia/cuda_stream.h>
namespace foreground::birefnet {
    void normalize_rgb(::cuda::stream_ref stream, tools::compute::TensorView output, const unsigned char* rgb);
    void sigmoid_mask(::cuda::stream_ref stream, unsigned char* output, tools::compute::TensorView input);
    void batch_norm(::cuda::stream_ref stream, tools::compute::TensorView output, tools::compute::TensorView input, tools::compute::TensorView weight, tools::compute::TensorView bias, tools::compute::TensorView mean, tools::compute::TensorView variance);
    void relu(::cuda::stream_ref stream, tools::compute::TensorView output, tools::compute::TensorView input);
    void resize(::cuda::stream_ref stream, tools::compute::TensorView output, tools::compute::TensorView input);
    void join(::cuda::stream_ref stream, tools::compute::TensorView output, tools::compute::TensorView input, int offset);
    void combine(::cuda::stream_ref stream, tools::compute::TensorView output, tools::compute::TensorView first, tools::compute::TensorView second, bool gate);
    void pool(::cuda::stream_ref stream, tools::compute::TensorView output, tools::compute::TensorView input);
    void patches(::cuda::stream_ref stream, tools::compute::TensorView output, tools::compute::TensorView input, bool merging);
    void windows(::cuda::stream_ref stream, tools::compute::TensorView output, tools::compute::TensorView input, int shift, bool reverse);
    void attention_bias(::cuda::stream_ref stream, tools::compute::TensorView output, tools::compute::TensorView table, int side, bool shifted);
    void deform_columns(::cuda::stream_ref stream, tools::compute::TensorView output, tools::compute::TensorView input, tools::compute::TensorView offsets, tools::compute::TensorView modulation, int kernel);
} // namespace foreground::birefnet
#endif
