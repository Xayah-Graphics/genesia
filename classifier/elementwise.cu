#include "kernel-common.cuh"
namespace classifier {
    __global__ void convert_kernel(Tensor out, Tensor in) {
        std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < out.elements()) write(out, i, read(in, i));
    }
    __global__ void add_kernel(Tensor out, Tensor a, Tensor b, float scale) {
        std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < out.elements()) write(out, i, read(a, i) + scale * read(b, i));
    }
    __global__ void gelu_kernel(Tensor out, Tensor x) {
        std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < x.elements()) {
            float v = read(x, i);
            write(out, i, .5f * v * (1.f + erff(v * .7071067811865475f)));
        }
    }
    __global__ void gelu_dx_kernel(Tensor dx, Tensor dy, Tensor x) {
        std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < x.elements()) {
            float v = read(x, i);
            write(dx, i, read(dy, i) * (.5f * (1.f + erff(v * .7071067811865475f)) + v * .3989422804014327f * expf(-.5f * v * v)));
        }
    }
    __global__ void mask_kernel(float* mask, RandomState* r, int n, float probability, int tag, bool training) {
        int i = threadIdx.x;
        if (i < n) mask[i] = !training || probability == 0 ? 1.f : (uniform(r, i, tag) < probability ? 0.f : rounded(1.f / (1.f - probability)));
    }
    __global__ void residual_kernel(Tensor out, Tensor branch, Tensor skip, const float* mask) {
        std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < out.elements()) write(out, i, rounded(read(branch, i) * mask[i / (std::size_t(out.h) * out.w * out.c)]) + read(skip, i));
    }
    __global__ void residual_dx_kernel(Tensor out, Tensor dy, const float* mask) {
        std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < out.elements()) write(out, i, read(dy, i) * mask[i / (std::size_t(out.h) * out.w * out.c)]);
    }
    __global__ void pool_kernel(Tensor out, Tensor x) {
        int c = blockIdx.x * 128 + threadIdx.x, n = blockIdx.y;
        if (c >= x.c) return;
        float sum = 0;
        for (int s = 0; s < x.h * x.w; ++s) sum += read(x, (std::size_t(n) * x.h * x.w + s) * x.c + c);
        write(out, n * x.c + c, sum / (x.h * x.w));
    }
    __global__ void pool_dx_kernel(Tensor dx, Tensor dy) {
        std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < dx.elements()) write(dx, i, read(dy, (i / (std::size_t(dx.h) * dx.w * dx.c)) * dx.c + i % dx.c) / (dx.h * dx.w));
    }
    __global__ void dropout_kernel(Tensor out, Tensor x, float* mask, RandomState* r, bool training) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= x.elements()) return;
        float m = !training ? 1.f : (uniform(r, i, 1000) < .1f ? 0.f : 1.f / .9f);
        mask[i] = m;
        write(out, i, read(x, i) * m);
    }
    __global__ void dropout_dx_kernel(Tensor dx, Tensor dy, const float* mask) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < dx.elements()) write(dx, i, read(dy, i) * mask[i]);
    }
    __global__ void bias_grad_kernel(Tensor dy, float* db) {
        int c = blockIdx.x * 128 + threadIdx.x;
        if (c >= dy.c) return;
        float sum = 0;
        for (int r = blockIdx.y; r < dy.n * dy.h * dy.w; r += gridDim.y) sum += read(dy, std::size_t(r) * dy.c + c);
        atomicAdd(db + c, sum);
    }
    __global__ void softmax_kernel(Tensor z, float* scores, int* decisions, unsigned char* accepted, float threshold) {
        int n         = blockIdx.x;
        float maximum = -INFINITY, sum = 0;
        for (int c = 0; c < z.c; ++c) maximum = fmaxf(maximum, read(z, n * z.c + c));
        for (int c = 0; c < z.c; ++c) sum += expf(read(z, n * z.c + c) - maximum);
        int other = 1;
        for (int c = 0; c < z.c; ++c) {
            scores[n * z.c + c] = expf(read(z, n * z.c + c) - maximum) / sum;
            if (c > 0 && read(z, n * z.c + c) > read(z, n * z.c + other)) other = c;
        }
        accepted[n]  = scores[n * z.c] >= threshold;
        decisions[n] = accepted[n] ? 0 : other;
    }
    __global__ void ce_kernel(Tensor dz, Tensor z, const int* labels, UpdateState* state, int effective) {
        int n         = blockIdx.x;
        float maximum = -INFINITY, sum = 0;
        for (int c = 0; c < z.c; ++c) maximum = fmaxf(maximum, read(z, n * z.c + c));
        for (int c = 0; c < z.c; ++c) sum += expf(read(z, n * z.c + c) - maximum);
        for (int c = 0; c < z.c; ++c) write(dz, n * z.c + c, (expf(read(z, n * z.c + c) - maximum) / sum - (c == labels[n])) / effective);
        atomicAdd(&state->loss, (logf(sum) + maximum - read(z, n * z.c + labels[n])) / effective);
    }
    void convert(cudaStream_t s, Tensor o, Tensor x) {
        convert_kernel<<<(o.elements() + 255) / 256, 256, 0, s>>>(o, x);
    }
    void add(cudaStream_t s, Tensor o, Tensor a, Tensor b, float scale) {
        add_kernel<<<(o.elements() + 255) / 256, 256, 0, s>>>(o, a, b, scale);
    }
    void gelu(cudaStream_t s, Tensor o, Tensor x) {
        gelu_kernel<<<(o.elements() + 255) / 256, 256, 0, s>>>(o, x);
    }
    void gelu_backward(cudaStream_t s, Tensor dx, Tensor dy, Tensor x) {
        gelu_dx_kernel<<<(dx.elements() + 255) / 256, 256, 0, s>>>(dx, dy, x);
    }
    void residual(cudaStream_t s, Tensor o, Tensor b, Tensor x, float p, float* mask, RandomState* r, int tag, bool train) {
        mask_kernel<<<1, 256, 0, s>>>(mask, r, x.n, p, tag, train);
        residual_kernel<<<(o.elements() + 255) / 256, 256, 0, s>>>(o, b, x, mask);
    }
    void residual_backward(cudaStream_t s, Tensor dx, Tensor dy, const float* mask) {
        residual_dx_kernel<<<(dx.elements() + 255) / 256, 256, 0, s>>>(dx, dy, mask);
    }
    void pool(cudaStream_t s, Tensor o, Tensor x) {
        pool_kernel<<<dim3((x.c + 127) / 128, x.n), 128, 0, s>>>(o, x);
    }
    void pool_backward(cudaStream_t s, Tensor dx, Tensor dy) {
        pool_dx_kernel<<<(dx.elements() + 255) / 256, 256, 0, s>>>(dx, dy);
    }
    void dropout(cudaStream_t s, Tensor o, Tensor x, float* mask, RandomState* r, bool train) {
        dropout_kernel<<<(x.elements() + 255) / 256, 256, 0, s>>>(o, x, mask, r, train);
    }
    void dropout_backward(cudaStream_t s, Tensor dx, Tensor dy, const float* mask) {
        dropout_dx_kernel<<<(dx.elements() + 255) / 256, 256, 0, s>>>(dx, dy, mask);
    }
    void bias_backward(cudaStream_t s, Tensor dy, float* db) {
        bias_grad_kernel<<<dim3((dy.c + 127) / 128, 128), 128, 0, s>>>(dy, db);
    }
    void softmax(cudaStream_t s, Tensor z, float* p, int* d, unsigned char* a, float t) {
        softmax_kernel<<<z.n, 1, 0, s>>>(z, p, d, a, t);
    }
    void cross_entropy(cudaStream_t s, Tensor dz, Tensor z, const int* labels, UpdateState* state, int effective) {
        ce_kernel<<<z.n, 1, 0, s>>>(dz, z, labels, state, effective);
    }
} // namespace classifier
