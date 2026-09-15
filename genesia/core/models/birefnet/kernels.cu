#include "kernels.h"
#include <cmath>
#include <cuda_fp16.h>
namespace genesia::birefnet {
    __global__ void normalize_rgb_kernel(__half* out, const unsigned char* rgb, int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        constexpr float mean[]{0.485F, 0.456F, 0.406F}, deviation[]{0.229F, 0.224F, 0.225F};
        out[i] = __float2half((float(rgb[i]) / 255 - mean[i % 3]) / deviation[i % 3]);
    }
    __global__ void sigmoid_mask_kernel(unsigned char* out, const __half* in, int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) out[i] = static_cast<unsigned char>(float(__float2half(1 / (1 + expf(-float(in[i]))))) * 255);
    }
    __global__ void batch_norm_kernel(__half* out, const __half* in, const __half* weight, const __half* bias, const __half* mean, const __half* variance, int count, int channels) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const int c = i % channels;
        out[i]      = __float2half((float(in[i]) - float(mean[c])) * rsqrtf(float(variance[c]) + 1.0e-5F) * float(weight[c]) + float(bias[c]));
    }
    __global__ void relu_kernel(__half* out, const __half* in, int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) out[i] = __float2half(fmaxf(0, float(in[i])));
    }
    __global__ void resize_kernel(__half* out, const __half* in, int oh, int ow, int ih, int iw, int c, int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const int channel = i % c, x = i / c % ow, y = i / c / ow % oh, batch = i / c / ow / oh;
        const float fx = ow > 1 ? float(x) * (iw - 1) / (ow - 1) : 0;
        const float fy = oh > 1 ? float(y) * (ih - 1) / (oh - 1) : 0;
        const int x0 = int(fx), y0 = int(fy), x1 = min(x0 + 1, iw - 1), y1 = min(y0 + 1, ih - 1);
        const float dx = fx - x0, dy = fy - y0;
        const auto* data = in + batch * ih * iw * c + channel;
        out[i]           = __float2half((1 - dy) * ((1 - dx) * float(data[(y0 * iw + x0) * c]) + dx * float(data[(y0 * iw + x1) * c])) + dy * ((1 - dx) * float(data[(y1 * iw + x0) * c]) + dx * float(data[(y1 * iw + x1) * c])));
    }
    __global__ void join_kernel(__half* out, const __half* in, int count, int oc, int ic, int offset) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) out[i / ic * oc + offset + i % ic] = in[i];
    }
    __global__ void combine_kernel(__half* out, const __half* a, const __half* b, int count, int channels, bool gate) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) out[i] = __float2half(gate ? float(a[i]) * float(__float2half(1 / (1 + expf(-float(b[i / channels]))))) : float(a[i]) + float(b[i]));
    }
    __global__ void pool_kernel(__half* out, const __half* in, int spatial, int channels) {
        __shared__ float sums[256];
        float sum = 0;
        for (int i = threadIdx.x; i < spatial; i += blockDim.x) sum += float(in[i * channels + blockIdx.x]);
        sums[threadIdx.x] = sum;
        __syncthreads();
        for (int s = 128; s; s /= 2) {
            if (threadIdx.x < s) sums[threadIdx.x] += sums[threadIdx.x + s];
            __syncthreads();
        }
        if (!threadIdx.x) out[blockIdx.x] = __float2half(sums[0] / spatial);
    }
    __global__ void patches_kernel(__half* out, const __half* in, int h, int w, int c, int ih, int iw, int ic, int count, bool merging) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const int channel = i % c, x = i / c % w, y = i / c / w;
        int sx, sy, sc;
        if (merging) {
            const int part = channel / ic;
            sx             = x * 2 + part / 2;
            sy             = y * 2 + part % 2;
            sc             = channel % ic;
        } else {
            const int gh = ih / h, gw = iw / w;
            sx = channel % gw * w + x;
            sy = channel / gw % gh * h + y;
            sc = channel / (gh * gw);
        }
        out[i] = sx < iw && sy < ih ? in[(sy * iw + sx) * ic + sc] : __float2half(0);
    }
    __global__ void windows_kernel(__half* out, const __half* in, int h, int w, int channels, int shift, bool reverse, int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const int hp = (h + 11) / 12 * 12, wp = (w + 11) / 12 * 12, c = i % channels;
        if (reverse) {
            const int y = (i / channels / w + hp - shift) % hp, x = (i / channels % w + wp - shift) % wp;
            const int window = y / 12 * (wp / 12) + x / 12, token = y % 12 * 12 + x % 12;
            out[i] = in[(window * 144 + token) * channels + c];
        } else {
            const int window = i / channels / 144, token = i / channels % 144;
            const int y = (window / (wp / 12) * 12 + token / 12 + shift) % hp, x = (window % (wp / 12) * 12 + token % 12 + shift) % wp;
            out[i] = y < h && x < w ? in[(y * w + x) * channels + c] : __float2half(0);
        }
    }
    __global__ void attention_bias_kernel(__half* out, const __half* table, int heads, int side, bool shifted, int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const int k = i % 144, q = i / 144 % 144, head = i / (144 * 144) % heads, window = i / (144 * 144 * heads);
        const int relative = (q / 12 - k / 12 + 11) * 23 + q % 12 - k % 12 + 11;
        float value        = float(table[relative * heads + head]);
        if (shifted) {
            const int yq = window / (side / 12) * 12 + q / 12, xq = window % (side / 12) * 12 + q % 12;
            const int yk = window / (side / 12) * 12 + k / 12, xk = window % (side / 12) * 12 + k % 12;
            const int rq = (yq < side - 12 ? 0 : yq < side - 6 ? 1 : 2) * 3 + (xq < side - 12 ? 0 : xq < side - 6 ? 1 : 2);
            const int rk = (yk < side - 12 ? 0 : yk < side - 6 ? 1 : 2) * 3 + (xk < side - 12 ? 0 : xk < side - 6 ? 1 : 2);
            if (rq != rk) value -= 100;
        }
        out[i] = __float2half(value);
    }
    __global__ void deform_columns_kernel(__half* out, const __half* in, const __half* offsets, const __half* modulation, int h, int w, int c, int kernel, int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const int channel = i % c, k = i / c % (kernel * kernel), pixel = i / c / (kernel * kernel);
        const float y = pixel / w - kernel / 2 + k / kernel + float(offsets[pixel * 2 * kernel * kernel + 2 * k]);
        const float x = pixel % w - kernel / 2 + k % kernel + float(offsets[pixel * 2 * kernel * kernel + 2 * k + 1]);
        float value   = 0;
        if (y > -1 && y < h && x > -1 && x < w) {
            const int y0 = int(floorf(y)), x0 = int(floorf(x));
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                    if (y0 + dy >= 0 && y0 + dy < h && x0 + dx >= 0 && x0 + dx < w) value += float(in[((y0 + dy) * w + x0 + dx) * c + channel]) * (dy ? y - y0 : 1 - y + y0) * (dx ? x - x0 : 1 - x + x0);
        }
        const float m = float(__float2half(2 * float(__float2half(1 / (1 + expf(-float(modulation[pixel * kernel * kernel + k])))))));
        out[i]        = __float2half(value * m);
    }
    void normalize_rgb(::cuda::stream_ref stream, compute::TensorView output, const unsigned char* rgb) {
        normalize_rgb_kernel<<<(output.elements() + 255) / 256, 256, 0, stream.get()>>>(static_cast<__half*>(output.data), rgb, int(output.elements()));
    }
    void sigmoid_mask(::cuda::stream_ref stream, unsigned char* output, compute::TensorView input) {
        sigmoid_mask_kernel<<<(input.elements() + 255) / 256, 256, 0, stream.get()>>>(output, static_cast<const __half*>(input.data), int(input.elements()));
    }
    void batch_norm(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input, compute::TensorView weight, compute::TensorView bias, compute::TensorView mean, compute::TensorView variance) {
        batch_norm_kernel<<<(output.elements() + 255) / 256, 256, 0, stream.get()>>>(static_cast<__half*>(output.data), static_cast<const __half*>(input.data), static_cast<const __half*>(weight.data), static_cast<const __half*>(bias.data), static_cast<const __half*>(mean.data), static_cast<const __half*>(variance.data), int(output.elements()), output.c);
    }
    void relu(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input) {
        relu_kernel<<<(output.elements() + 255) / 256, 256, 0, stream.get()>>>(static_cast<__half*>(output.data), static_cast<const __half*>(input.data), int(output.elements()));
    }
    void resize(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input) {
        resize_kernel<<<(output.elements() + 255) / 256, 256, 0, stream.get()>>>(static_cast<__half*>(output.data), static_cast<const __half*>(input.data), output.h, output.w, input.h, input.w, input.c, int(output.elements()));
    }
    void join(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input, int offset) {
        join_kernel<<<(input.elements() + 255) / 256, 256, 0, stream.get()>>>(static_cast<__half*>(output.data), static_cast<const __half*>(input.data), int(input.elements()), output.c, input.c, offset);
    }
    void combine(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView first, compute::TensorView second, bool gate) {
        combine_kernel<<<(output.elements() + 255) / 256, 256, 0, stream.get()>>>(static_cast<__half*>(output.data), static_cast<const __half*>(first.data), static_cast<const __half*>(second.data), int(output.elements()), first.c, gate);
    }
    void pool(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input) {
        pool_kernel<<<input.c, 256, 0, stream.get()>>>(static_cast<__half*>(output.data), static_cast<const __half*>(input.data), input.h * input.w, input.c);
    }
    void patches(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input, bool merging) {
        patches_kernel<<<(output.elements() + 255) / 256, 256, 0, stream.get()>>>(static_cast<__half*>(output.data), static_cast<const __half*>(input.data), output.h, output.w, output.c, input.h, input.w, input.c, int(output.elements()), merging);
    }
    void windows(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input, int shift, bool reverse) {
        windows_kernel<<<(output.elements() + 255) / 256, 256, 0, stream.get()>>>(static_cast<__half*>(output.data), static_cast<const __half*>(input.data), reverse ? output.h : input.h, reverse ? output.w : input.w, output.c, shift, reverse, int(output.elements()));
    }
    void attention_bias(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView table, int side, bool shifted) {
        attention_bias_kernel<<<(output.elements() + 255) / 256, 256, 0, stream.get()>>>(static_cast<__half*>(output.data), static_cast<const __half*>(table.data), output.h, side, shifted, int(output.elements()));
    }
    void deform_columns(::cuda::stream_ref stream, compute::TensorView output, compute::TensorView input, compute::TensorView offsets, compute::TensorView modulation, int kernel) {
        deform_columns_kernel<<<(output.elements() + 255) / 256, 256, 0, stream.get()>>>(static_cast<__half*>(output.data), static_cast<const __half*>(input.data), static_cast<const __half*>(offsets.data), static_cast<const __half*>(modulation.data), input.h, input.w, input.c, kernel, int(output.elements()));
    }
} // namespace genesia::birefnet
