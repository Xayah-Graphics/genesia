module;
#include "../../compute/inference-kernels.h"
#include "../../compute/tensor.h"
#include "kernels.h"
#include <cuda_fp16.h>
#include <genesia/cuda.h>

module genesia.models.sdxl.unet;
import std;
import genesia.compute.inference;
import genesia.models.sdxl.workspace;
import genesia.models.sdxl.weights;

namespace genesia::sdxl {
    Residual::Residual(Checkpoint& source, const std::string& prefix, const bool timed) {
        const compute::Scalar type = timed ? compute::Scalar::f16 : compute::Scalar::bf16;
        norm1                      = source.norm(prefix + (timed ? ".in_layers.0" : ".norm1"), type, timed ? 1.0e-5F : 1.0e-6F);
        conv1                      = source.convolution(prefix + (timed ? ".in_layers.2" : ".conv1"), type);
        if (timed) time = source.linear(prefix + ".emb_layers.1", type);
        norm2 = source.norm(prefix + (timed ? ".out_layers.0" : ".norm2"), type, timed ? 1.0e-5F : 1.0e-6F);
        conv2 = source.convolution(prefix + (timed ? ".out_layers.3" : ".conv2"), type);
        if (conv1.weight.c != conv1.weight.n) shortcut = source.convolution(prefix + (timed ? ".skip_connection" : ".nin_shortcut"), type, 1, 0);
    }

    void Residual::forward(const compute::TensorView output, const compute::TensorView input, const compute::TensorView projected_time, const int* step, compute::InferenceRuntime& runtime, const Workspace& scratch, const bool prepared_statistics) const {
        const compute::TensorView normalized = scratch.normalized.reshape(input.n, input.h, input.w, input.c, input.scalar);
        const compute::TensorView hidden     = scratch.hidden.reshape(input.n, input.h, input.w, output.c, input.scalar);
        const compute::TensorView activated  = scratch.normalized.reshape(input.n, input.h, input.w, output.c, input.scalar);
        runtime.group_norm(normalized, input, norm1, scratch.statistics, true, prepared_statistics);
        runtime.convolution(hidden, normalized, conv1);
        runtime.group_norm(activated, hidden, norm2, scratch.statistics, true, false, projected_time, step);
        compute::TensorView skip = input;
        if (shortcut.weight.data) {
            skip = scratch.shortcut.reshape(input.n, input.h, input.w, output.c, input.scalar);
            runtime.convolution(skip, input, shortcut);
        }
        runtime.convolution(output, activated, conv2, skip);
    }

    Transformer::Transformer(Checkpoint& source, const std::string& prefix) {
        norm1        = source.norm(prefix + ".norm1", compute::Scalar::f16);
        qkv          = source.qkv(std::array{prefix + ".attn1.to_q", prefix + ".attn1.to_k", prefix + ".attn1.to_v"}, compute::Scalar::f16, false);
        self_output  = source.linear(prefix + ".attn1.to_out.0", compute::Scalar::f16);
        norm2        = source.norm(prefix + ".norm2", compute::Scalar::f16);
        query        = source.linear(prefix + ".attn2.to_q", compute::Scalar::f16, false);
        kv           = source.qkv(std::array{prefix + ".attn2.to_k", prefix + ".attn2.to_v"}, compute::Scalar::f16, false);
        cross_output = source.linear(prefix + ".attn2.to_out.0", compute::Scalar::f16);
        norm3        = source.norm(prefix + ".norm3", compute::Scalar::f16);
        expand       = source.linear(prefix + ".ff.net.0.proj", compute::Scalar::f16);
        contract     = source.linear(prefix + ".ff.net.2", compute::Scalar::f16);
    }

    void Transformer::forward(const compute::TensorView values, const compute::TensorView cross_key, const compute::TensorView cross_value, const std::int32_t* lengths, compute::InferenceRuntime& runtime, const Workspace& scratch) const {
        const int width                      = values.c;
        const compute::TensorView normalized = scratch.normalized.reshape(values.n, values.h, values.w, width);
        const compute::TensorView packed     = scratch.hidden.reshape(values.n, values.h, values.w, width * 3);
        const compute::TensorView attended   = scratch.output.reshape(values.n, values.h, values.w, width);
        runtime.layer_norm(normalized, values, norm1);
        runtime.linear(packed, normalized, qkv);
        compute::TensorView k = packed;
        compute::TensorView v = packed;
        k.data                = static_cast<std::byte*>(packed.data) + width * 2;
        v.data                = static_cast<std::byte*>(packed.data) + width * 4;
        runtime.attention(attended, packed, k, v, width / 64, width * 3, width * 3);
        runtime.linear(normalized, attended, self_output);
        compute::kernels::residual_norm(runtime.stream, normalized.data, values.data, normalized.data, norm2.weight.data, norm2.bias.data, values.n * values.h * values.w, width, norm2.epsilon);
        const compute::TensorView q = scratch.hidden.reshape(values.n, values.h, values.w, width);
        runtime.linear(q, normalized, query);
        runtime.attention(attended, q, cross_key, cross_value, width / 64, width, width * 2, false, nullptr, lengths);
        runtime.linear(normalized, attended, cross_output);
        compute::kernels::residual_norm(runtime.stream, normalized.data, values.data, normalized.data, norm3.weight.data, norm3.bias.data, values.n * values.h * values.w, width, norm3.epsilon);
        const compute::TensorView activated = scratch.output.reshape(values.n, values.h, values.w, width * 4);
        runtime.geglu(activated, normalized, expand);
        runtime.linear(values, activated, contract, values);
    }

    SpatialTransformer::SpatialTransformer(Checkpoint& source, const std::string& prefix, const int depth) {
        norm  = source.norm(prefix + ".norm", compute::Scalar::f16, 1.0e-6F);
        input = source.linear(prefix + ".proj_in", compute::Scalar::f16);
        for (int i = 0; i < depth; ++i) blocks.emplace_back(source, prefix + ".transformer_blocks." + std::to_string(i));
        output = source.linear(prefix + ".proj_out", compute::Scalar::f16);
    }

    void SpatialTransformer::forward(const compute::TensorView values, compute::TensorView sequence, const std::span<const std::array<compute::TensorView, 2>> context, const std::int32_t* lengths, compute::InferenceRuntime& runtime, const Workspace& scratch) const {
        const compute::TensorView normalized = scratch.normalized.reshape(values.n, values.h, values.w, values.c);
        sequence.n                           = values.n;
        sequence.h                           = values.h;
        sequence.w                           = values.w;
        sequence.c                           = values.c;
        runtime.group_norm(normalized, values, norm, scratch.statistics, false);
        runtime.linear(sequence, normalized, input);
        for (std::size_t i = 0; i < blocks.size(); ++i) blocks[i].forward(sequence, context[i][0], context[i][1], lengths, runtime, scratch);
        runtime.linear(values, sequence, output, values);
    }

    UNetState::UNetState(const ::cuda::stream_ref stream, const int h, const int w) : lengths{stream, ::cuda::device_default_memory_pool(stream.device()), 2, ::cuda::no_init}, height{h}, width{w} {
        skips.emplace_back(stream, ::cuda::device_default_memory_pool(stream.device()), 2uz * h * w * 320, ::cuda::no_init);
        for (int level = 0; level < 3; ++level) {
            for (int i = 0; i < 2; ++i) skips.emplace_back(stream, ::cuda::device_default_memory_pool(stream.device()), 2uz * (h >> level) * (w >> level) * (320 << level), ::cuda::no_init);
            if (level < 2) skips.emplace_back(stream, ::cuda::device_default_memory_pool(stream.device()), 2uz * (h >> (level + 1)) * (w >> (level + 1)) * (320 << level), ::cuda::no_init);
        }
    }

    UNet::UNet(Checkpoint& source) {
        const std::string root = "model.diffusion_model.";
        time_input             = source.linear(root + "time_embed.0", compute::Scalar::f16);
        time_output            = source.linear(root + "time_embed.2", compute::Scalar::f16);
        label_input            = source.linear(root + "label_emb.0.0", compute::Scalar::f16);
        label_output           = source.linear(root + "label_emb.0.2", compute::Scalar::f16);
        input                  = source.convolution(root + "input_blocks.0.0", compute::Scalar::f16);
        const std::array<int, 3> depths{0, 2, 10};
        int index = 1;
        for (int level = 0; level < 3; ++level) {
            for (int i = 0; i < 2; ++i, ++index) {
                const std::string prefix = root + "input_blocks." + std::to_string(index);
                down[level].residuals.emplace_back(source, prefix + ".0", true);
                if (depths[level]) down[level].transformers.emplace_back(source, prefix + ".1", depths[level]);
            }
            if (level < 2) down[level].resize = source.convolution(root + "input_blocks." + std::to_string(index++) + ".0.op", compute::Scalar::f16, 2);
        }
        middle_input  = Residual{source, root + "middle_block.0", true};
        middle        = SpatialTransformer{source, root + "middle_block.1", 10};
        middle_output = Residual{source, root + "middle_block.2", true};
        index         = 0;
        for (int level = 0; level < 3; ++level) {
            for (int i = 0; i < 3; ++i, ++index) {
                const std::string prefix = root + "output_blocks." + std::to_string(index);
                up[level].residuals.emplace_back(source, prefix + ".0", true);
                if (depths[2 - level]) up[level].transformers.emplace_back(source, prefix + ".1", depths[2 - level]);
                if (i == 2 && level < 2) up[level].resize = source.convolution(prefix + ".2.conv", compute::Scalar::f16);
            }
        }
        norm   = source.norm(root + "out.0", compute::Scalar::f16);
        output = source.convolution(root + "out.2", compute::Scalar::f16);
    }

    void UNet::bind(const std::map<const void*, void*>& weights) {
        const auto matrix = [&](compute::TensorView& view) {
            if (const auto found = weights.find(view.data); found != weights.end()) view.data = found->second;
        };
        const auto residual = [&](Residual& block) {
            for (auto* view : {&block.conv1.weight, &block.conv2.weight, &block.time.weight, &block.shortcut.weight}) matrix(*view);
        };
        const auto transformer = [&](SpatialTransformer& block) {
            matrix(block.input.weight);
            for (auto& layer : block.blocks)
                for (auto* view : {&layer.qkv.weight, &layer.self_output.weight, &layer.query.weight, &layer.kv.weight, &layer.cross_output.weight, &layer.expand.weight, &layer.contract.weight}) matrix(*view);
            matrix(block.output.weight);
        };
        for (auto* view : {&time_input.weight, &time_output.weight, &label_input.weight, &label_output.weight, &input.weight, &output.weight}) matrix(*view);
        for (auto* stages : {&down, &up})
            for (auto& stage : *stages) {
                for (auto& block : stage.residuals) residual(block);
                for (auto& block : stage.transformers) transformer(block);
                matrix(stage.resize.weight);
            }
        residual(middle_input);
        transformer(middle);
        residual(middle_output);
    }

    void UNet::prepare(UNetCondition& state, const compute::TensorView context, const compute::TensorView condition, const float* times, const int steps, compute::InferenceRuntime& runtime, const Workspace& scratch) const {
        auto embeddings                         = ::cuda::device_buffer<__half>{runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), steps * 320uz, ::cuda::no_init};
        const compute::TensorView expanded      = scratch.hidden.reshape(steps, 1, 1, 1280);
        const compute::TensorView time          = scratch.output.reshape(steps, 1, 1, 1280);
        const compute::TensorView label         = scratch.hidden.reshape(2, 1, 1, 1280);
        const compute::TensorView labels_hidden = scratch.normalized.reshape(2, 1, 1, 1280);
        const compute::TensorView summed        = scratch.combined.reshape(steps * 2, 1, 1, 1280);
        kernels::time_embedding(runtime.stream, embeddings.data(), times, steps, 320, 1);
        runtime.linear(expanded, {embeddings.data(), steps, 1, 1, 320}, time_input);
        compute::kernels::activation(runtime.stream, expanded.data, expanded.data, expanded.elements(), 0);
        runtime.linear(time, expanded, time_output);
        runtime.linear(labels_hidden, condition, label_input);
        compute::kernels::activation(runtime.stream, labels_hidden.data, labels_hidden.data, labels_hidden.elements(), 0);
        runtime.linear(label, labels_hidden, label_output);
        kernels::time_condition(runtime.stream, summed.data, time.data, label.data, steps);
        const auto prepare_residual = [&](const Residual& residual) {
            const int width = residual.time.weight.w;
            auto& storage   = state.time_storage.emplace_back(runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), steps * 2uz * width, ::cuda::no_init);
            const compute::TensorView projected{storage.data(), steps * 2, 1, 1, width};
            runtime.linear(projected, summed, residual.time);
            state.time.push_back(projected);
        };
        const auto prepare_transformer = [&](const SpatialTransformer& transformer) {
            for (const auto& block : transformer.blocks) {
                const int width = block.kv.weight.w / 2;
                auto& storage   = state.context_storage.emplace_back(runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), context.n * context.h * context.w * std::size_t(width) * 2, ::cuda::no_init);
                const compute::TensorView packed{storage.data(), context.n, context.h, context.w, width * 2};
                runtime.linear(packed, context, block.kv);
                const std::array pair{compute::TensorView{storage.data(), context.n, context.h, context.w, width}, compute::TensorView{storage.data() + width, context.n, context.h, context.w, width}};
                state.context.push_back(pair);
            }
        };
        for (const auto& stage : down) {
            for (const auto& residual : stage.residuals) prepare_residual(residual);
            for (const auto& transformer : stage.transformers) prepare_transformer(transformer);
        }
        prepare_residual(middle_input);
        prepare_transformer(middle);
        prepare_residual(middle_output);
        for (const auto& stage : up) {
            for (const auto& residual : stage.residuals) prepare_residual(residual);
            for (const auto& transformer : stage.transformers) prepare_transformer(transformer);
        }
    }

    void UNet::forward(const compute::TensorView result, const compute::TensorView model_input, const int* step, UNetState& state, const UNetCondition& condition, compute::InferenceRuntime& runtime, const Workspace& scratch) const {
        compute::TensorView current{state.skips[0].data(), 2, state.height, state.width, 320};
        const compute::TensorView sequence{state.sequence.data(), 2, 1, 1, 1};
        std::size_t skip_index{};
        std::size_t time_index{};
        std::size_t context_index{};
        std::array<compute::TensorView, 9> skips;
        const auto save_skip = [&] {
            skips[skip_index] = current;
            ++skip_index;
        };
        const auto transform = [&](const SpatialTransformer& transformer) {
            transformer.forward(current, sequence, std::span{condition.context}.subspan(context_index, transformer.blocks.size()), state.lengths.data(), runtime, scratch);
            context_index += transformer.blocks.size();
        };
        runtime.convolution(current, model_input, input);
        save_skip();
        for (const auto& stage : down) {
            for (std::size_t i = 0; i < stage.residuals.size(); ++i) {
                compute::TensorView next = current;
                next.data                = state.skips[skip_index].data();
                next.c                   = stage.residuals[i].conv1.weight.n;
                stage.residuals[i].forward(next, current, condition.time[time_index++], step, runtime, scratch);
                current = next;
                if (!stage.transformers.empty()) transform(stage.transformers[i]);
                save_skip();
            }
            if (stage.resize.weight.data) {
                const compute::TensorView resized{state.skips[skip_index].data(), 2, current.h / 2, current.w / 2, current.c};
                runtime.convolution(resized, current, stage.resize);
                current = resized;
                save_skip();
            }
        }
        const compute::TensorView middle_result{state.current.data(), 2, current.h, current.w, current.c};
        middle_input.forward(middle_result, current, condition.time[time_index++], step, runtime, scratch);
        current = middle_result;
        transform(middle);
        middle_output.forward(current, current, condition.time[time_index++], step, runtime, scratch);
        for (const auto& stage : up) {
            for (std::size_t i = 0; i < stage.residuals.size(); ++i) {
                const compute::TensorView skip   = skips[--skip_index];
                const compute::TensorView joined = scratch.combined.reshape(2, current.h, current.w, current.c + skip.c);
                compute::kernels::group_moments(runtime.stream, scratch.statistics, current.data, skip.data, joined.data, nullptr, nullptr, 2, current.h * current.w, current.c + skip.c, current.c, 1);
                current.c = stage.residuals[i].conv1.weight.n;
                stage.residuals[i].forward(current, joined, condition.time[time_index++], step, runtime, scratch, true);
                if (!stage.transformers.empty()) transform(stage.transformers[i]);
            }
            if (stage.resize.weight.data) {
                const compute::TensorView resized = scratch.combined.reshape(2, current.h * 2, current.w * 2, current.c);
                compute::kernels::resize(runtime.stream, resized.data, current.data, 2, current.h, current.w, current.c, 1);
                current.h *= 2;
                current.w *= 2;
                runtime.convolution(current, resized, stage.resize);
            }
        }
        const compute::TensorView activated = scratch.normalized.reshape(2, current.h, current.w, current.c);
        runtime.group_norm(activated, current, norm, scratch.statistics, true);
        runtime.convolution(result, activated, output);
    }
} // namespace genesia::sdxl
