module;
#include "../neural/inference-kernels.h"
#include "kernels.h"
#include <genesia/cuda.h>

module genesia.sdxl.vae;
import std;
import genesia.neural.inference_runtime;
import genesia.sdxl.workspace;
import genesia.sdxl.weights;
import genesia.sdxl.unet;

namespace genesia::sdxl {
    VaeAttention::VaeAttention(Checkpoint& source, const std::string& prefix) : norm{source.norm(prefix + ".norm", neural::Scalar::bf16, 1.0e-6F)}, qkv{source.qkv(std::array{prefix + ".q", prefix + ".k", prefix + ".v"}, neural::Scalar::bf16, true)}, projection{source.convolution(prefix + ".proj_out", neural::Scalar::bf16, 1, 0)} {}

    void VaeAttention::forward(const neural::TensorView current, neural::InferenceRuntime& runtime, const Workspace& scratch) const {
        const neural::TensorView normalized = scratch.normalized.reshape(1, current.h, current.w, 512, neural::Scalar::bf16);
        const neural::TensorView packed     = scratch.hidden.reshape(1, current.h, current.w, 1536, neural::Scalar::bf16);
        const neural::TensorView attended   = scratch.output.reshape(1, current.h, current.w, 512, neural::Scalar::bf16);
        runtime.group_norm(normalized, current, norm, scratch.statistics, false);
        runtime.linear(packed, normalized, qkv);
        auto key   = packed;
        auto value = packed;
        key.data   = static_cast<std::byte*>(packed.data) + 1024;
        value.data = static_cast<std::byte*>(packed.data) + 2048;
        runtime.attention(attended, packed, key, value, 1, 1536, 1536);
        runtime.convolution(current, attended, projection, current);
    }

    VAEEncoder::VAEEncoder(Checkpoint& source) : attention{source, "first_stage_model.encoder.mid.attn_1"} {
        const std::string root = "first_stage_model.encoder.";
        input                  = source.convolution(root + "conv_in", neural::Scalar::bf16);
        for (int level = 0; level < 4; ++level) {
            const auto prefix = root + "down." + std::to_string(level);
            for (int i = 0; i < 2; ++i) down[level].blocks.emplace_back(source, prefix + ".block." + std::to_string(i), false);
            if (level < 3) down[level].resize = source.convolution(prefix + ".downsample.conv", neural::Scalar::bf16, 2, 0);
        }
        middle_input  = Residual{source, root + "mid.block_1", false};
        middle_output = Residual{source, root + "mid.block_2", false};
        norm          = source.norm(root + "norm_out", neural::Scalar::bf16, 1.0e-6F);
        output        = source.convolution(root + "conv_out", neural::Scalar::bf16);
        quant         = source.convolution("first_stage_model.quant_conv", neural::Scalar::bf16, 1, 0);
    }

    void VAEEncoder::forward(const neural::TensorView latent, const neural::TensorView pixels, neural::TensorView current, neural::InferenceRuntime& runtime, const Workspace& scratch) const {
        const auto normalized = scratch.normalized.reshape(1, pixels.h, pixels.w, 3, neural::Scalar::bf16);
        kernels::image_encode(runtime.stream, normalized.data, static_cast<const std::uint8_t*>(pixels.data), pixels.h * pixels.w * 3);
        current = current.reshape(1, pixels.h, pixels.w, 128, neural::Scalar::bf16);
        runtime.convolution(current, normalized, input);
        for (const auto& stage : down) {
            for (const auto& block : stage.blocks) {
                auto next = current;
                next.c    = block.conv1.weight.n;
                block.forward(next, current, {}, nullptr, runtime, scratch);
                current = next;
            }
            if (stage.resize.weight.data) {
                const auto padded = scratch.combined.reshape(1, current.h + 1, current.w + 1, current.c, neural::Scalar::bf16);
                kernels::encoder_pad(runtime.stream, padded.data, current.data, current.h, current.w, current.c);
                current.h /= 2;
                current.w /= 2;
                runtime.convolution(current, padded, stage.resize);
            }
        }
        middle_input.forward(current, current, {}, nullptr, runtime, scratch);
        attention.forward(current, runtime, scratch);
        middle_output.forward(current, current, {}, nullptr, runtime, scratch);
        const auto activated = scratch.normalized.reshape(1, current.h, current.w, 512, neural::Scalar::bf16);
        const auto moments   = scratch.hidden.reshape(1, current.h, current.w, 8, neural::Scalar::bf16);
        const auto quantized = scratch.output.reshape(1, current.h, current.w, 8, neural::Scalar::bf16);
        runtime.group_norm(activated, current, norm, scratch.statistics, true);
        runtime.convolution(moments, activated, output);
        runtime.convolution(quantized, moments, quant);
        kernels::latent_encode(runtime.stream, static_cast<float*>(latent.data), quantized.data, static_cast<int>(latent.elements()));
    }

    VAE::VAE(Checkpoint& source) : attention{source, "first_stage_model.decoder.mid.attn_1"} {
        const std::string root = "first_stage_model.decoder.";
        post_quant             = source.convolution("first_stage_model.post_quant_conv", neural::Scalar::bf16, 1, 0);
        input                  = source.convolution(root + "conv_in", neural::Scalar::bf16);
        middle_input           = Residual{source, root + "mid.block_1", false};
        middle_output          = Residual{source, root + "mid.block_2", false};
        for (int level = 0; level < 4; ++level) {
            const std::string prefix = root + "up." + std::to_string(3 - level);
            for (int i = 0; i < 3; ++i) up[level].blocks.emplace_back(source, prefix + ".block." + std::to_string(i), false);
            if (level < 3) up[level].resize = source.convolution(prefix + ".upsample.conv", neural::Scalar::bf16);
        }
        norm   = source.norm(root + "norm_out", neural::Scalar::bf16, 1.0e-6F);
        output = source.convolution(root + "conv_out", neural::Scalar::bf16);
    }

    void VAE::forward(const neural::TensorView result, const neural::TensorView latent, neural::TensorView current, neural::InferenceRuntime& runtime, const Workspace& scratch) const {
        const neural::TensorView scaled    = scratch.normalized.reshape(1, latent.h, latent.w, 4, neural::Scalar::bf16);
        const neural::TensorView quantized = scratch.output.reshape(1, latent.h, latent.w, 4, neural::Scalar::bf16);
        kernels::latent_decode(runtime.stream, scaled.data, static_cast<const float*>(latent.data), static_cast<int>(latent.elements()));
        runtime.convolution(quantized, scaled, post_quant);
        current.n      = 1;
        current.h      = latent.h;
        current.w      = latent.w;
        current.c      = 512;
        current.scalar = neural::Scalar::bf16;
        runtime.convolution(current, quantized, input);
        middle_input.forward(current, current, {}, nullptr, runtime, scratch);
        attention.forward(current, runtime, scratch);
        middle_output.forward(current, current, {}, nullptr, runtime, scratch);
        for (const auto& stage : up) {
            for (const auto& block : stage.blocks) {
                neural::TensorView next = current;
                next.c                  = block.conv1.weight.n;
                block.forward(next, current, {}, nullptr, runtime, scratch);
                current = next;
            }
            if (stage.resize.weight.data) {
                const neural::TensorView resized = scratch.combined.reshape(1, current.h * 2, current.w * 2, current.c, neural::Scalar::bf16);
                neural::kernels::resize(runtime.stream, resized.data, current.data, 1, current.h, current.w, current.c, 2);
                current.h *= 2;
                current.w *= 2;
                runtime.convolution(current, resized, stage.resize);
            }
        }
        const neural::TensorView activated = scratch.normalized.reshape(1, current.h, current.w, 128, neural::Scalar::bf16);
        runtime.group_norm(activated, current, norm, scratch.statistics, true);
        runtime.convolution(result, activated, output);
    }
} // namespace genesia::sdxl
