#include "repaint-kernels.h"
#include <cmath>
#include <cuda/launch>
#include <genesia/cuda.h>

namespace genesia::repaint_kernels {
    namespace {
        __device__ float linear(const std::uint8_t value) {
            const float c = float(value) / 255.0F;
            return c <= 0.04045F ? c / 12.92F : powf((c + 0.055F) / 1.055F, 2.4F);
        }

        __device__ std::uint8_t srgb(float value) {
            value = fminf(fmaxf(value, 0.0F), 1.0F);
            const float c = value <= 0.0031308F ? value * 12.92F : 1.055F * powf(value, 1.0F / 2.4F) - 0.055F;
            return static_cast<std::uint8_t>(__float2int_rn(c * 255.0F));
        }

        __device__ float lanczos(const float distance) {
            const float x = fabsf(distance);
            if (x < 1.0e-5F) return 1.0F;
            if (x >= 3.0F) return 0.0F;
            const float p = 3.141592653589793F * x;
            return sinf(p) * sinf(p / 3.0F) * 3.0F / (p * p);
        }

        __global__ void resize_x_kernel(float* output, const std::uint8_t* input, const int source_width, const int x, const int y, const int width, const int height, const int target_width) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= target_width * height * 3) return;
            const int row = i / (target_width * 3);
            const int col = i / 3 % target_width;
            const int channel = i % 3;
            if (width == target_width) {
                output[i] = linear(input[((row + y) * source_width + col + x) * 3 + channel]);
                return;
            }
            const float ratio = float(width) / target_width;
            const float filter = fmaxf(1.0F, ratio);
            const float center = (col + 0.5F) * ratio - 0.5F;
            float sum = 0.0F, weights = 0.0F;
            for (int j = int(ceilf(center - 3.0F * filter)); j <= int(floorf(center + 3.0F * filter)); ++j) {
                const float weight = lanczos((j - center) / filter);
                const int source = ((row + y) * source_width + min(max(j, 0), width - 1) + x) * 3 + channel;
                sum = fmaf(weight, linear(input[source]), sum);
                weights += weight;
            }
            output[i] = sum / weights;
        }

        __global__ void resize_work_kernel(std::uint8_t* output, const float* horizontal, const Geometry g) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= g.work_width * g.work_height * 3) return;
            const int x = min(i / 3 % g.work_width, g.resized_width - 1);
            const int y = min(i / (g.work_width * 3), g.resized_height - 1);
            const int channel = i % 3;
            if (g.crop_height == g.resized_height) {
                output[i] = srgb(horizontal[(y * g.resized_width + x) * 3 + channel]);
                return;
            }
            const float ratio = float(g.crop_height) / g.resized_height;
            const float filter = fmaxf(1.0F, ratio);
            const float center = (y + 0.5F) * ratio - 0.5F;
            float sum = 0.0F, weights = 0.0F;
            for (int j = int(ceilf(center - 3.0F * filter)); j <= int(floorf(center + 3.0F * filter)); ++j) {
                const float weight = lanczos((j - center) / filter);
                sum = fmaf(weight, horizontal[(min(max(j, 0), g.crop_height - 1) * g.resized_width + x) * 3 + channel], sum);
                weights += weight;
            }
            output[i] = srgb(sum / weights);
        }

        __global__ void mask_kernel(std::uint8_t* output, const std::uint8_t* mask, const Geometry g) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= g.work_width / 8 * (g.work_height / 8)) return;
            const int x = i % (g.work_width / 8) * 8;
            const int y = i / (g.work_width / 8) * 8;
            // Coverage, rather than point sampling, keeps narrow defects from disappearing.
            const int x0 = g.x + min(int(float(x) * g.crop_width / g.resized_width), g.crop_width - 1);
            const int y0 = g.y + min(int(float(y) * g.crop_height / g.resized_height), g.crop_height - 1);
            const int x1 = g.x + min(int(ceilf(float(x + 8) * g.crop_width / g.resized_width)), g.crop_width);
            const int y1 = g.y + min(int(ceilf(float(y + 8) * g.crop_height / g.resized_height)), g.crop_height);
            std::uint8_t covered = 0;
            for (int row = y0; row < y1 && !covered; ++row)
                for (int col = x0; col < x1; ++col) covered |= mask[row * g.width + col];
            output[i] = covered;
        }

        __global__ void alpha_kernel(float* output, const std::uint8_t* mask, const int width, const int height, const int feather) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= width * height) return;
            if (!mask[i]) {
                output[i] = 0.0F;
                return;
            }
            if (!feather) {
                output[i] = 1.0F;
                return;
            }
            const int x = i % width, y = i / width;
            int distance = feather * feather;
            for (int dy = -feather; dy <= feather; ++dy)
                for (int dx = -feather; dx <= feather; ++dx) {
                    const int squared = dx * dx + dy * dy;
                    if (squared >= distance || x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height) continue;
                    if (!mask[(y + dy) * width + x + dx]) distance = squared;
                }
            const float t = sqrtf(float(distance)) / feather;
            output[i] = t * t * (3.0F - 2.0F * t);
        }

        __global__ void composite_kernel(std::uint8_t* output, const float* horizontal, const std::uint8_t* original, const float* alpha, const Geometry g) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= g.width * g.height * 3) return;
            const float blend = alpha[i / 3];
            if (blend == 0.0F) {
                output[i] = original[i];
                return;
            }
            const int x = i / 3 % g.width - g.x;
            const int y = i / (g.width * 3) - g.y;
            const int channel = i % 3;
            float value;
            if (g.resized_height == g.crop_height) value = horizontal[(y * g.crop_width + x) * 3 + channel];
            else {
                const float ratio = float(g.resized_height) / g.crop_height;
                const float filter = fmaxf(1.0F, ratio);
                const float center = (y + 0.5F) * ratio - 0.5F;
                float sum = 0.0F, weights = 0.0F;
                for (int j = int(ceilf(center - 3.0F * filter)); j <= int(floorf(center + 3.0F * filter)); ++j) {
                    const float weight = lanczos((j - center) / filter);
                    sum = fmaf(weight, horizontal[(min(max(j, 0), g.resized_height - 1) * g.crop_width + x) * 3 + channel], sum);
                    weights += weight;
                }
                value = sum / weights;
            }
            output[i] = srgb(blend == 1.0F ? value : fmaf(blend, value - linear(original[i]), linear(original[i])));
        }
    } // namespace

    void prepare(const ::cuda::stream_ref stream, std::uint8_t* work, std::uint8_t* latent_mask, float* alpha, float* scratch, const std::uint8_t* original, const std::uint8_t* mask, const Geometry g, const int feather) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((g.resized_width * g.crop_height * 3 + 255) / 256), ::cuda::block_dims(256))), resize_x_kernel, scratch, original, g.width, g.x, g.y, g.crop_width, g.crop_height, g.resized_width);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((g.work_width * g.work_height * 3 + 255) / 256), ::cuda::block_dims(256))), resize_work_kernel, work, scratch, g);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((g.work_width / 8 * (g.work_height / 8) + 255) / 256), ::cuda::block_dims(256))), mask_kernel, latent_mask, mask, g);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((g.width * g.height + 255) / 256), ::cuda::block_dims(256))), alpha_kernel, alpha, mask, g.width, g.height, feather);
    }

    void composite(const ::cuda::stream_ref stream, std::uint8_t* output, float* scratch, const std::uint8_t* work, const std::uint8_t* original, const float* alpha, const Geometry g) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((g.crop_width * g.resized_height * 3 + 255) / 256), ::cuda::block_dims(256))), resize_x_kernel, scratch, work, g.work_width, 0, 0, g.resized_width, g.resized_height, g.crop_width);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((g.width * g.height * 3 + 255) / 256), ::cuda::block_dims(256))), composite_kernel, output, scratch, original, alpha, g);
    }
} // namespace genesia::repaint_kernels
