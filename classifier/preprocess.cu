#include "kernel-common.cuh"
namespace classifier {
    __global__ void advance_kernel(RandomState* r) {
        ++r->sequence;
    }
    __global__ void augment_parameters(Augment* a, RandomState* r, int n) {
        int i = threadIdx.x;
        if (i < n) a[i] = {.92f + .16f * uniform(r, i, 1), .92f + .16f * uniform(r, i, 2), uniform(r, i, 3) < .5f, uniform(r, i, 4) < .5f, 0};
    }
    __device__ int blend(int x, float factor, int mean) {
        return max(0, min(255, int(mean + factor * (x - mean))));
    }
    __global__ void augment_mean(const unsigned char* rgb, std::size_t stride, std::size_t image_stride, int h, int w, Augment* a) {
        __shared__ cub::BlockReduce<unsigned long long, 256>::TempStorage shared;
        unsigned long long sum = 0;
        int n                  = blockIdx.x;
        for (int i = threadIdx.x; i < h * w; i += blockDim.x) {
            const unsigned char* p = rgb + n * image_stride + (i / w) * stride + (i % w) * 3;
            int r = p[0], g = p[1], b = p[2];
            if (a[n].order == 0) {
                r = blend(r, a[n].brightness, 0);
                g = blend(g, a[n].brightness, 0);
                b = blend(b, a[n].brightness, 0);
            }
            sum += (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16;
        }
        sum = cub::BlockReduce<unsigned long long, 256>(shared).Sum(sum);
        if (threadIdx.x == 0) a[n].mean = floorf(float(sum) / float(h * w) + .5f);
    }
    __global__ void preprocess_kernel(Tensor out, const unsigned char* rgb, std::size_t stride, std::size_t image_stride, const Augment* a, bool training) {
        std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= out.elements()) return;
        int c = i % 3, x = (i / 3) % out.w, y = (i / (3 * out.w)) % out.h, n = i / (std::size_t(out.h) * out.w * 3);
        int sx    = training && a[n].flip ? out.w - 1 - x : x;
        int value = rgb[n * image_stride + y * stride + sx * 3 + c];
        if (training) {
            if (a[n].order == 0) value = blend(blend(value, a[n].brightness, 0), a[n].contrast, int(a[n].mean));
            else value = blend(blend(value, a[n].contrast, int(a[n].mean)), a[n].brightness, 0);
        }
        constexpr float mean[3] = {.485f, .456f, .406f}, sd[3] = {.229f, .224f, .225f};
        write(out, i, ((float(value) / 255.f) - mean[c]) / sd[c]);
    }
    void preprocess(cudaStream_t s, Tensor o, const unsigned char* rgb, std::size_t stride, std::size_t image_stride, Augment* a, RandomState* r, bool train) {
        if (train) {
            augment_parameters<<<1, 256, 0, s>>>(a, r, o.n);
            augment_mean<<<o.n, 256, 0, s>>>(rgb, stride, image_stride, o.h, o.w, a);
        }
        preprocess_kernel<<<(o.elements() + 255) / 256, 256, 0, s>>>(o, rgb, stride, image_stride, a, train);
    }
    void random_advance(cudaStream_t s, RandomState* r) {
        advance_kernel<<<1, 1, 0, s>>>(r);
    }
} // namespace classifier
