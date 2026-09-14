#ifndef CLASSIFIER_KERNELS_H
#define CLASSIFIER_KERNELS_H
#include "../../compute/tensor.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
namespace cudnn_frontend::graph {
    class Graph;
}
namespace genesia::convnext {
    struct RandomState {
        std::uint64_t seed = 42, sequence = 0;
    };
    struct Augment {
        float brightness, contrast;
        int flip, order;
        float mean;
    };
    struct UpdateState {
        int step = 0, schedule_steps = 400, head_only_steps = 25, effective_batch = 64, warmup_steps = 10;
        float norm2 = 0, loss = 0, clip = 1;
        float head_only_lr = .001f, backbone_lr = .00003f, head_lr = .0003f, weight_decay = .01f, clip_norm = 1;
    };
    struct ParameterView {
        float* master;
        void* bf16;
        float* grad;
        float* m;
        float* v;
        std::size_t count;
        bool decay, head;
    };
    void convert(cudaStream_t stream, compute::TensorView out, compute::TensorView in);
    void add(cudaStream_t stream, compute::TensorView out, compute::TensorView a, compute::TensorView b, float scale = 1.f);
    void preprocess(cudaStream_t stream, compute::TensorView out, const unsigned char* rgb, std::size_t stride, std::size_t image_stride, Augment* augment, RandomState* random, bool training);
    void random_advance(cudaStream_t stream, RandomState* random);
    void layer_norm(cudaStream_t stream, compute::TensorView out, compute::TensorView x, const float* weight, const float* bias, float* stats);
    void layer_norm_backward(cudaStream_t stream, compute::TensorView dx, compute::TensorView dy, compute::TensorView x, const float* weight, const float* stats, float* dw, float* db, float* scratch);
    void depthwise(cudaStream_t stream, compute::TensorView out, compute::TensorView x, const void* weight, const void* bias);
    void depthwise_backward(cudaStream_t stream, compute::TensorView dx, compute::TensorView dy, compute::TensorView x, const void* weight, float* dw, float* db, float* scratch);
    void gelu(cudaStream_t stream, compute::TensorView out, compute::TensorView x);
    void gelu_backward(cudaStream_t stream, compute::TensorView dx, compute::TensorView dy, compute::TensorView x);
    void grn(cudaStream_t stream, compute::TensorView out, compute::TensorView x, const float* weight, const float* bias, float* stats, float* scratch);
    void grn_backward(cudaStream_t stream, compute::TensorView dx, compute::TensorView dy, compute::TensorView x, const float* weight, const float* stats, float* dw, float* db, float* scratch);
    void residual(cudaStream_t stream, compute::TensorView out, compute::TensorView branch, compute::TensorView skip, float probability, float* mask, RandomState* random, int tag, bool training);
    void residual_backward(cudaStream_t stream, compute::TensorView branch_grad, compute::TensorView dy, const float* mask);
    void pool(cudaStream_t stream, compute::TensorView out, compute::TensorView x);
    void pool_backward(cudaStream_t stream, compute::TensorView dx, compute::TensorView dy);
    void dropout(cudaStream_t stream, compute::TensorView out, compute::TensorView x, float* mask, RandomState* random, bool training);
    void dropout_backward(cudaStream_t stream, compute::TensorView dx, compute::TensorView dy, const float* mask);
    void bias_backward(cudaStream_t stream, compute::TensorView dy, float* db);
    void softmax(cudaStream_t stream, compute::TensorView logits, float* scores, int* decisions);
    void cross_entropy(cudaStream_t stream, compute::TensorView dz, compute::TensorView logits, const int* labels, UpdateState* state, int effective_batch);
    void gradient_norm(cudaStream_t stream, ParameterView p, UpdateState* state);
    void optimizer_begin(cudaStream_t stream, UpdateState* state);
    void optimizer_clip(cudaStream_t stream, UpdateState* state);
    void adamw(cudaStream_t stream, ParameterView p, UpdateState* state);
} // namespace genesia::convnext
#endif
