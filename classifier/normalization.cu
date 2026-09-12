#include "kernel-common.cuh"
namespace classifier {
    __global__ void ln_kernel(Tensor out, Tensor x, const float* weight, const float* bias, float* stats) {
        int lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32, c = x.c;
        if (row >= x.n * x.h * x.w) return;
        float origin = read(x, std::size_t(row) * c), sum = 0;
        for (int j = lane; j < c; j += 32) sum += read(x, std::size_t(row) * c + j) - origin;
#pragma unroll
        for (int offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        float mean = origin + __shfl_sync(0xffffffff, sum, 0) / c;
        sum        = 0;
        for (int j = lane; j < c; j += 32) {
            float v = read(x, std::size_t(row) * c + j) - mean;
            sum += v * v;
        }
#pragma unroll
        for (int offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / c + 1e-6f);
        if (lane == 0 && stats) {
            stats[2 * row]     = mean;
            stats[2 * row + 1] = inv;
        }
        for (int j = lane; j < c; j += 32) write(out, std::size_t(row) * c + j, (read(x, std::size_t(row) * c + j) - mean) * inv * weight[j] + bias[j]);
    }
    __global__ void ln_dx_kernel(Tensor dx, Tensor dy, Tensor x, const float* weight, const float* stats) {
        __shared__ cub::BlockReduce<float2, 256>::TempStorage shared;
        __shared__ float2 sums;
        int row = blockIdx.x, c = x.c;
        float mean = stats[2 * row], inv = stats[2 * row + 1];
        float2 a = {0, 0};
        for (int j = threadIdx.x; j < c; j += 256) {
            std::size_t i = std::size_t(row) * c + j;
            float g       = read(dy, i) * weight[j];
            a.x += g;
            a.y += g * (read(x, i) - mean) * inv;
        }
        a = cub::BlockReduce<float2, 256>(shared).Reduce(a, [] __device__(float2 p, float2 q) { return make_float2(p.x + q.x, p.y + q.y); });
        if (threadIdx.x == 0) sums = make_float2(a.x / c, a.y / c);
        __syncthreads();
        for (int j = threadIdx.x; j < c; j += 256) {
            std::size_t i = std::size_t(row) * c + j;
            write(dx, i, inv * (read(dy, i) * weight[j] - sums.x - (read(x, i) - mean) * inv * sums.y));
        }
    }
    __global__ void ln_param_kernel(Tensor dy, Tensor x, const float* stats, float* dw, float* db) {
        int c = blockIdx.x * 128 + threadIdx.x;
        if (c >= x.c) return;
        float a = 0, b = 0;
        int rows = x.n * x.h * x.w;
        for (int r = blockIdx.y; r < rows; r += gridDim.y) {
            std::size_t i = std::size_t(r) * x.c + c;
            float g       = read(dy, i);
            a += g * (read(x, i) - stats[2 * r]) * stats[2 * r + 1];
            b += g;
        }
        atomicAdd(dw + c, a);
        atomicAdd(db + c, b);
    }
    __global__ void grn_partial(Tensor x, float* scratch) {
        int c = blockIdx.x * 128 + threadIdx.x;
        if (c >= x.c) return;
        int n = blockIdx.y, parts = gridDim.z, part = blockIdx.z, spatial = x.h * x.w;
        float v = 0;
        for (int s = part; s < spatial; s += parts) {
            float a = read(x, (std::size_t(n) * spatial + s) * x.c + c);
            v       = fmaf(a, a, v);
        }
        scratch[(n * parts + part) * x.c + c] = v;
    }
    __global__ void grn_stats(Tensor x, const float* partial, float* stats, int parts) {
        __shared__ cub::BlockReduce<float, 256>::TempStorage shared;
        __shared__ float denominator;
        int n = blockIdx.x, c = x.c;
        float sum = 0;
        for (int j = threadIdx.x; j < c; j += 256) {
            float v = 0;
            for (int p = 0; p < parts; ++p) v += partial[(n * parts + p) * c + j];
            v                      = sqrtf(v);
            stats[(n * c + j) * 2] = v;
            sum += v;
        }
        sum = cub::BlockReduce<float, 256>(shared).Sum(sum);
        if (threadIdx.x == 0) denominator = sum / c + 1e-6f;
        __syncthreads();
        for (int j = threadIdx.x; j < c; j += 256) stats[(n * c + j) * 2 + 1] = stats[(n * c + j) * 2] / denominator;
        if (threadIdx.x == 0) stats[2 * x.n * c + n] = denominator;
    }
    __global__ void grn_apply(Tensor out, Tensor x, const float* weight, const float* bias, const float* stats) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= x.elements()) return;
        int c = i % x.c, n = x.n == 1 ? 0 : i / (x.h * x.w * x.c);
        float v = read(x, i), xn = v * stats[(n * x.c + c) * 2 + 1];
        write(out, i, v + fmaf(weight[c], xn, bias[c]));
    }
    __global__ void grn_backward_partial(Tensor dy, Tensor x, const float* stats, float* dw, float* db, float* partial) {
        int c = blockIdx.x * 128 + threadIdx.x;
        if (c >= x.c) return;
        int n = blockIdx.y, part = blockIdx.z, parts = gridDim.z, spatial = x.h * x.w;
        float dot = 0, a = 0, b = 0;
        for (int s = part; s < spatial; s += parts) {
            std::size_t i = (std::size_t(n) * spatial + s) * x.c + c;
            float d = read(dy, i), v = read(x, i);
            dot += d * v;
            a += d * v * stats[(n * x.c + c) * 2 + 1];
            b += d;
        }
        partial[(n * parts + part) * x.c + c] = dot;
        atomicAdd(dw + c, a);
        atomicAdd(db + c, b);
    }
    __global__ void grn_backward_stats(Tensor x, const float* weight, const float* stats, float* partial, int parts) {
        __shared__ cub::BlockReduce<float, 256>::TempStorage shared;
        __shared__ float total;
        int n = blockIdx.x, c = x.c;
        float sum = 0, den = stats[2 * x.n * c + n];
        float* result = partial + x.n * parts * c;
        for (int j = threadIdx.x; j < c; j += 256) {
            float dot = 0;
            for (int p = 0; p < parts; ++p) dot += partial[(n * parts + p) * c + j];
            result[n * c + j] = dot * weight[j];
            sum += dot * weight[j] * stats[(n * c + j) * 2];
        }
        sum = cub::BlockReduce<float, 256>(shared).Sum(sum);
        if (threadIdx.x == 0) total = sum / (c * den * den);
        __syncthreads();
        for (int j = threadIdx.x; j < c; j += 256) {
            float g           = stats[(n * c + j) * 2];
            result[n * c + j] = g > 0 ? (result[n * c + j] / den - total) / g : 0;
        }
    }
    __global__ void grn_dx(Tensor dx, Tensor dy, Tensor x, const float* weight, const float* stats, const float* correction) {
        std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= x.elements()) return;
        int c = i % x.c, n = i / (std::size_t(x.h) * x.w * x.c);
        write(dx, i, read(dy, i) * (1 + weight[c] * stats[(n * x.c + c) * 2 + 1]) + read(x, i) * correction[n * x.c + c]);
    }
    void layer_norm(cudaStream_t s, Tensor o, Tensor x, const float* w, const float* b, float* stats) {
        ln_kernel<<<(x.n * x.h * x.w + 3) / 4, 128, 0, s>>>(o, x, w, b, stats);
    }
    void layer_norm_backward(cudaStream_t s, Tensor dx, Tensor dy, Tensor x, const float* w, const float* stats, float* dw, float* db, float*) {
        ln_dx_kernel<<<x.n * x.h * x.w, 256, 0, s>>>(dx, dy, x, w, stats);
        ln_param_kernel<<<dim3((x.c + 127) / 128, 128), 128, 0, s>>>(dy, x, stats, dw, db);
    }
    void grn(cudaStream_t s, Tensor o, Tensor x, const float* w, const float* b, float* stats, float* scratch) {
        grn_partial<<<dim3((x.c + 127) / 128, x.n, 128), 128, 0, s>>>(x, scratch);
        grn_stats<<<x.n, 256, 0, s>>>(x, scratch, stats, 128);
        grn_apply<<<(x.elements() + 255) / 256, 256, 0, s>>>(o, x, w, b, stats);
    }
    void grn_backward(cudaStream_t s, Tensor dx, Tensor dy, Tensor x, const float* w, const float* stats, float* dw, float* db, float* scratch) {
        grn_backward_partial<<<dim3((x.c + 127) / 128, x.n, 128), 128, 0, s>>>(dy, x, stats, dw, db, scratch);
        grn_backward_stats<<<x.n, 256, 0, s>>>(x, w, stats, scratch, 128);
        grn_dx<<<(dx.elements() + 255) / 256, 256, 0, s>>>(dx, dy, x, w, stats, scratch + x.n * 128 * x.c);
    }
} // namespace classifier
