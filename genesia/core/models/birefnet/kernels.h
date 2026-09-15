#ifndef GENESIA_BIREFNET_KERNELS_H
#define GENESIA_BIREFNET_KERNELS_H
#include "../../compute/tensor.h"
#include <genesia/cuda_stream.h>
namespace genesia::birefnet {
    void normalize_rgb(::cuda::stream_ref stream, compute::TensorView output, const unsigned char* rgb);
    void sigmoid_mask(::cuda::stream_ref stream, unsigned char* output, compute::TensorView input);
    void batch_norm(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input, compute::TensorView weight, compute::TensorView bias, compute::TensorView mean, compute::TensorView variance);
    void relu(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input);
    void resize(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input);
    void join(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input, int offset);
    void combine(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView first, compute::TensorView second, bool gate);
    void pool(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input);
    void patches(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input, bool merging);
    void windows(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input, int shift, bool reverse);
    void attention_bias(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView table, int side, bool shifted);
    void deform_columns(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input, compute::TensorView offsets, compute::TensorView modulation, int kernel);
} // namespace genesia::birefnet
#endif
