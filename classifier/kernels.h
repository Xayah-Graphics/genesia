#ifndef CLASSIFIER_KERNELS_H
#define CLASSIFIER_KERNELS_H
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
namespace cudnn_frontend::graph {
    class Graph;
}
namespace classifier {
    struct Tensor {
        void* data = nullptr;
        int n = 1, h = 1, w = 1, c = 1;
        bool fp32 = false;
        __host__ __device__ std::size_t elements() const {
            return std::size_t(n) * h * w * c;
        }
        __host__ __device__ std::size_t bytes() const {
            return elements() * (fp32 ? 4 : 2);
        }
    };
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
    void convert(cudaStream_t stream, Tensor out, Tensor in);
    void add(cudaStream_t stream, Tensor out, Tensor a, Tensor b, float scale = 1.f);
    void preprocess(cudaStream_t stream, Tensor out, const unsigned char* rgb, std::size_t stride, std::size_t image_stride, Augment* augment, RandomState* random, bool training);
    void random_advance(cudaStream_t stream, RandomState* random);
    void layer_norm(cudaStream_t stream, Tensor out, Tensor x, const float* weight, const float* bias, float* stats);
    void layer_norm_backward(cudaStream_t stream, Tensor dx, Tensor dy, Tensor x, const float* weight, const float* stats, float* dw, float* db, float* scratch);
    void depthwise(cudaStream_t stream, Tensor out, Tensor x, const void* weight, const void* bias);
    void depthwise_backward(cudaStream_t stream, Tensor dx, Tensor dy, Tensor x, const void* weight, float* dw, float* db, float* scratch);
    void gelu(cudaStream_t stream, Tensor out, Tensor x);
    void gelu_backward(cudaStream_t stream, Tensor dx, Tensor dy, Tensor x);
    void grn(cudaStream_t stream, Tensor out, Tensor x, const float* weight, const float* bias, float* stats, float* scratch);
    void grn_backward(cudaStream_t stream, Tensor dx, Tensor dy, Tensor x, const float* weight, const float* stats, float* dw, float* db, float* scratch);
    void residual(cudaStream_t stream, Tensor out, Tensor branch, Tensor skip, float probability, float* mask, RandomState* random, int tag, bool training);
    void residual_backward(cudaStream_t stream, Tensor branch_grad, Tensor dy, const float* mask);
    void pool(cudaStream_t stream, Tensor out, Tensor x);
    void pool_backward(cudaStream_t stream, Tensor dx, Tensor dy);
    void dropout(cudaStream_t stream, Tensor out, Tensor x, float* mask, RandomState* random, bool training);
    void dropout_backward(cudaStream_t stream, Tensor dx, Tensor dy, const float* mask);
    void bias_backward(cudaStream_t stream, Tensor dy, float* db);
    void softmax(cudaStream_t stream, Tensor logits, float* scores, int* decisions, unsigned char* accepted, float threshold);
    void cross_entropy(cudaStream_t stream, Tensor dz, Tensor logits, const int* labels, UpdateState* state, int effective_batch);
    void gradient_norm(cudaStream_t stream, ParameterView p, UpdateState* state);
    void optimizer_begin(cudaStream_t stream, UpdateState* state);
    void optimizer_clip(cudaStream_t stream, UpdateState* state);
    void adamw(cudaStream_t stream, ParameterView p, UpdateState* state);
} // namespace classifier
#endif
