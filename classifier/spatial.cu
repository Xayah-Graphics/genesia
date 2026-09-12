#include "kernel-common.cuh"
namespace classifier {
    template <int Channels>
    __global__ void dw_packed(Tensor out, const __nv_bfloat162* x, const __nv_bfloat162* weight, const __nv_bfloat162* bias) {
        std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= out.elements() / 2) return;
        constexpr int pairs = Channels / 2;
        int c = i % pairs, w = (i / pairs) % out.w, h = (i / (pairs * out.w)) % out.h;
        float2 sum = {0, 0};
#pragma unroll
        for (int y = 0; y < 7; ++y) {
            int iy = h + y - 3;
#pragma unroll
            for (int z = 0; z < 7; ++z) {
                int ix = w + z - 3;
                if (iy >= 0 && iy < out.h && ix >= 0 && ix < out.w) {
                    float2 a = __bfloat1622float2(x[i + ((y - 3) * out.w + z - 3) * pairs]), b = __bfloat1622float2(weight[(y * 7 + z) * pairs + c]);
                    sum.x = fmaf(a.x, b.x, sum.x);
                    sum.y = fmaf(a.y, b.y, sum.y);
                }
            }
        }
        float2 b                                  = __bfloat1622float2(bias[c]);
        static_cast<__nv_bfloat162*>(out.data)[i] = __floats2bfloat162_rn(rounded(sum.x) + b.x, rounded(sum.y) + b.y);
    }
    __global__ void dw_dx_kernel(Tensor dx, Tensor dy, const void* weight) {
        std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= dx.elements()) return;
        int c = i % dx.c, w = (i / dx.c) % dx.w, h = (i / (dx.c * dx.w)) % dx.h, n = i / (std::size_t(dx.c) * dx.w * dx.h);
        float sum = 0;
#pragma unroll
        for (int y = 0; y < 7; ++y) {
            int iy = h + 3 - y;
#pragma unroll
            for (int z = 0; z < 7; ++z) {
                int ix = w + 3 - z;
                if (iy >= 0 && iy < dx.h && ix >= 0 && ix < dx.w) sum = fmaf(read(dy, ((std::size_t(n) * dx.h + iy) * dx.w + ix) * dx.c + c), bf(weight, (y * 7 + z) * dx.c + c), sum);
            }
        }
        write(dx, i, sum);
    }
    __global__ void dw_param_kernel(Tensor dy, Tensor x, float* dw, float* db) {
        int c = blockIdx.x * 128 + threadIdx.x;
        if (c >= x.c) return;
        int k = blockIdx.y, y = k / 7 - 3, z = k % 7 - 3, rows = x.n * x.h * x.w;
        float sum = 0, bias = 0;
        for (int r = blockIdx.z; r < rows; r += gridDim.z) {
            int ix = r % x.w + z, iy = (r / x.w) % x.h + y, n = r / (x.w * x.h);
            float d = read(dy, std::size_t(r) * x.c + c);
            bias += d;
            if (ix >= 0 && ix < x.w && iy >= 0 && iy < x.h) sum = fmaf(d, rounded(read(x, ((std::size_t(n) * x.h + iy) * x.w + ix) * x.c + c)), sum);
        }
        atomicAdd(dw + k * x.c + c, sum);
        if (k == 0) atomicAdd(db + c, bias);
    }
    void depthwise(cudaStream_t s, Tensor o, Tensor x, const void* w, const void* b) {
        auto input = static_cast<const __nv_bfloat162*>(x.data), weight = static_cast<const __nv_bfloat162*>(w), bias = static_cast<const __nv_bfloat162*>(b);
        int blocks = int((o.elements() / 2 + 255) / 256);
        switch (o.c) {
        case 96: dw_packed<96><<<blocks, 256, 0, s>>>(o, input, weight, bias); break;
        case 192: dw_packed<192><<<blocks, 256, 0, s>>>(o, input, weight, bias); break;
        case 384: dw_packed<384><<<blocks, 256, 0, s>>>(o, input, weight, bias); break;
        case 768: dw_packed<768><<<blocks, 256, 0, s>>>(o, input, weight, bias); break;
        }
    }
    void depthwise_backward(cudaStream_t s, Tensor dx, Tensor dy, Tensor x, const void* w, float* dw, float* db, float*) {
        dw_dx_kernel<<<(dx.elements() + 255) / 256, 256, 0, s>>>(dx, dy, w);
        dw_param_kernel<<<dim3((x.c + 127) / 128, 49, 64), 128, 0, s>>>(dy, x, dw, db);
    }
} // namespace classifier
