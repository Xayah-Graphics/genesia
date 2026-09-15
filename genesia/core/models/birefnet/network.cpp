module;
#include "../../compute/inference-kernels.h"
#include "kernels.h"
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>
module genesia.models.birefnet;
import genesia.project;
import std;
namespace genesia::birefnet {
    Network::Network(const std::filesystem::path& model) : stream{::cuda::devices[0]}, runtime{stream, std::filesystem::path{project::cache} / "birefnet"}, operator_workspace{stream, ::cuda::device_default_memory_pool(stream.device())} {
        files::SafeFile file{model};
        tensors.push_back({nullptr, 1, 1024, 1024, 3});
        auto features    = backbone(file, 0);
        const auto small = backbone(file, append(file, Operation::resize, {0}, {}, 512));
        for (int i = 0; i < 4; ++i) features[i] = append(file, Operation::join, {features[i], append(file, Operation::resize, {small[i]}, {}, tensors[features[i]].h)});
        std::vector<int> context;
        for (int i = 0; i < 3; ++i) context.push_back(append(file, Operation::resize, {features[i]}, {}, 32));
        context.push_back(features[3]);
        int x = decoder_block(file, append(file, Operation::join, std::move(context)), "squeeze_module.0");
        for (int level = 4; level >= 1; --level) {
            const int detail = input_block(file, tensors[x].h, level + 1);
            x                = decoder_block(file, append(file, Operation::join, {x, detail}), std::format("decoder.decoder_block{}", level));
            if (level > 1) {
                int gate          = append(file, Operation::convolution, {x}, std::format("decoder.gdt_convs_{}.0", level));
                gate              = append(file, Operation::batchnorm, {gate}, std::format("decoder.gdt_convs_{}.1", level));
                gate              = append(file, Operation::relu, {gate});
                gate              = append(file, Operation::convolution, {gate}, std::format("decoder.gdt_convs_attn_{}.0", level));
                x                 = append(file, Operation::gate, {x, gate});
                const int lateral = append(file, Operation::convolution, {features[level - 2]}, std::format("decoder.lateral_block{}.conv", level));
                x                 = append(file, Operation::add, {append(file, Operation::resize, {x}, {}, tensors[lateral].h), lateral});
            }
        }
        x = append(file, Operation::resize, {x}, {}, 1024);
        x = append(file, Operation::join, {x, input_block(file, 1024, 1)});
        append(file, Operation::convolution, {x}, "decoder.conv_out1.0");

        // Assign storage from exact tensor lifetimes; no activations are retained for training.
        std::vector<std::size_t> last(tensors.size()), offsets(tensors.size());
        for (std::size_t i = 0; i < nodes.size(); ++i)
            for (int input : nodes[i].inputs) last[input] = i + 1;
        last.back() = tensors.size();
        std::vector<std::vector<std::size_t>> releases(tensors.size() + 1);
        std::map<std::size_t, std::size_t> free;
        std::size_t bytes{};
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            for (const auto old : releases[i]) {
                std::size_t offset = offsets[old], size = (tensors[old].bytes() + 255) & ~255uz;
                auto next = free.lower_bound(offset);
                if (next != free.end() && offset + size == next->first) {
                    size += next->second;
                    next = free.erase(next);
                }
                if (next != free.begin()) {
                    const auto previous = std::prev(next);
                    if (previous->first + previous->second == offset) {
                        offset = previous->first;
                        size += previous->second;
                        free.erase(previous);
                    }
                }
                free.emplace(offset, size);
            }
            const auto size = (tensors[i].bytes() + 255) & ~255uz;
            auto fit        = free.end();
            for (auto it = free.begin(); it != free.end(); ++it)
                if (it->second >= size && (fit == free.end() || it->second < fit->second)) fit = it;
            if (fit == free.end()) {
                offsets[i] = bytes;
                bytes += size;
            } else {
                offsets[i] = fit->first;
                if (fit->second > size) free.emplace(fit->first + size, fit->second - size);
                free.erase(fit);
            }
            if (last[i] + 1 < releases.size()) releases[last[i] + 1].push_back(i);
        }
        arena  = compute::DeviceBuffer{bytes};
        pixels = compute::DeviceBuffer{1024 * 1024 * 3};
        mask   = compute::DeviceBuffer{1024 * 1024};
        for (std::size_t i = 0; i < tensors.size(); ++i) tensors[i].data = static_cast<std::byte*>(arena.data) + offsets[i];
        compute::check(cudaMemsetAsync(tensors.front().data, 0, tensors.front().bytes(), stream.get()));
        forward();
        operator_workspace = runtime.finish_preparation();
        compute::check(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
        try {
            forward();
        } catch (...) {
            cudaGraph_t unfinished{};
            cudaStreamEndCapture(stream.get(), &unfinished);
            if (unfinished) cudaGraphDestroy(unfinished);
            throw;
        }
        compute::check(cudaStreamEndCapture(stream.get(), std::out_ptr(graph)));
        compute::check(cudaGraphInstantiate(std::out_ptr(executable), graph.get(), 0));
        stream.sync();
    }
    Network::~Network() {
        stream.sync();
    }
    std::vector<std::uint8_t> Network::infer(const std::span<const std::uint8_t> rgb) {
        compute::check(cudaMemcpyAsync(pixels.data, rgb.data(), pixels.size, cudaMemcpyHostToDevice, stream.get()));
        normalize_rgb(stream, tensors.front(), static_cast<const unsigned char*>(pixels.data));
        compute::check(cudaGraphLaunch(executable.get(), stream.get()));
        sigmoid_mask(stream, static_cast<unsigned char*>(mask.data), tensors.back());
        std::vector<std::uint8_t> result(mask.size);
        compute::check(cudaMemcpyAsync(result.data(), mask.data, result.size(), cudaMemcpyDeviceToHost, stream.get()));
        stream.sync();
        return result;
    }
    compute::TensorView Network::parameter(files::SafeFile& file, const std::string& name) {
        if (const auto found = parameters.find(name); found != parameters.end()) return found->second;
        const auto& entry  = file.header.at(name);
        const auto shape   = entry.at("shape").get<std::vector<int>>();
        const auto offsets = entry.at("data_offsets").get<std::array<std::size_t, 2>>();
        const auto count   = std::accumulate(shape.begin(), shape.end(), 1uz, std::multiplies<>{});
        const auto type    = entry.at("dtype").get<std::string>();
        static const std::map<std::string, compute::Scalar> types{{"F32", compute::Scalar::f32}, {"F16", compute::Scalar::f16}};
        compute::DeviceBuffer source{offsets[1] - offsets[0]};
        compute::check(cudaMemcpyAsync(source.data, file.base + offsets[0], source.size, cudaMemcpyHostToDevice, stream.get()));
        auto& storage = weights.emplace_back(count * 2);
        compute::TensorView result{storage.data, 1, 1, shape.size() == 1 ? 1 : shape[0], shape.size() == 1 ? shape[0] : shape[1]};
        if (shape.size() == 4) {
            result = {storage.data, shape[0], shape[2], shape[3], shape[1]};
            compute::kernels::convert_layout(stream, result.data, source.data, result.n, result.h * result.w, result.c, int(types.at(type)), int(compute::Scalar::f16));
        } else compute::kernels::convert(stream, result.data, source.data, count, int(types.at(type)), int(compute::Scalar::f16));
        stream.sync();
        parameters.emplace(name, result);
        return result;
    }
    int Network::append(files::SafeFile& file, const Operation operation, std::vector<int> inputs, std::string name, const int argument, const int second) {
        Node node{operation, std::move(inputs), {}, argument, second};
        auto output = tensors[node.inputs.front()];
        output.data = nullptr;
        switch (operation) {
        case Operation::convolution:
        case Operation::linear:
            {
                auto weight       = parameter(file, name + ".weight");
                const bool biased = operation == Operation::convolution ? !name.ends_with("dec_att.conv1") && !name.ends_with("global_avg_pool.1") : !name.ends_with(".reduction") && !name.ends_with(".regular_conv");
                if (biased) node.weights[1] = parameter(file, name + ".bias");
                if (operation == Operation::convolution) {
                    node.argument = argument ? argument : 1;
                    node.second   = name == "bb.patch_embed.proj" ? 0 : weight.h / 2;
                    output.h      = (output.h + 2 * node.second - weight.h) / node.argument + 1;
                    output.w      = (output.w + 2 * node.second - weight.w) / node.argument + 1;
                    output.c      = weight.n;
                    if (!biased) {
                        auto& zero = weights.emplace_back(output.c * 2);
                        compute::check(cudaMemsetAsync(zero.data, 0, zero.size, stream.get()));
                        node.weights[1] = {zero.data, 1, 1, 1, output.c};
                    }
                } else {
                    if (name.ends_with(".regular_conv")) weight = weight.reshape(1, 1, weight.n, weight.h * weight.w * weight.c);
                    output.c = weight.w;
                }
                node.weights[0] = weight;
                break;
            }
        case Operation::norm:
        case Operation::batchnorm:
            node.weights[0] = parameter(file, name + ".weight");
            node.weights[1] = parameter(file, name + ".bias");
            if (operation == Operation::batchnorm) {
                node.weights[2] = parameter(file, name + ".running_mean");
                node.weights[3] = parameter(file, name + ".running_var");
            }
            break;
        case Operation::resize: output.h = output.w = argument; break;
        case Operation::join:
            output.c = 0;
            for (int input : node.inputs) output.c += tensors[input].c;
            break;
        case Operation::pool: output.h = output.w = 1; break;
        case Operation::patches:
            output.c *= output.h / argument * (output.w / argument);
            output.h = output.w = argument;
            break;
        case Operation::merge:
            output.h = (output.h + 1) / 2;
            output.w = (output.w + 1) / 2;
            output.c *= 4;
            break;
        case Operation::windows:
            output.n = (output.h + 11) / 12 * ((output.w + 11) / 12);
            output.h = 1;
            output.w = 144;
            break;
        case Operation::unwindows:
            output.n = 1;
            output.h = output.w = argument;
            break;
        case Operation::bias:
            node.weights[0] = parameter(file, name);
            output.h        = output.c / 32;
            output.w = output.c = 144;
            break;
        case Operation::attention: output.c /= 3; break;
        case Operation::deform: output.c *= argument * argument; break;
        default: break;
        }
        nodes.push_back(std::move(node));
        tensors.push_back(output);
        return int(tensors.size() - 1);
    }
    std::array<int, 4> Network::backbone(files::SafeFile& file, const int input) {
        int x = append(file, Operation::convolution, {input}, "bb.patch_embed.proj", 4);
        x     = append(file, Operation::norm, {x}, "bb.patch_embed.norm");
        std::array<int, 4> outputs;
        constexpr std::array depths{2, 2, 18, 2};
        for (int stage = 0; stage < 4; ++stage) {
            for (int block = 0; block < depths[stage]; ++block) {
                const auto prefix = std::format("bb.layers.{}.blocks.{}", stage, block);
                const int side = tensors[x].h, shift = block % 2 * 6;
                int branch     = append(file, Operation::norm, {x}, prefix + ".norm1");
                branch         = append(file, Operation::windows, {branch}, {}, shift);
                const int bias = append(file, Operation::bias, {branch}, prefix + ".attn.relative_position_bias_table", (side + 11) / 12 * 12, shift);
                branch         = append(file, Operation::linear, {branch}, prefix + ".attn.qkv");
                branch         = append(file, Operation::attention, {branch, bias});
                branch         = append(file, Operation::linear, {branch}, prefix + ".attn.proj");
                branch         = append(file, Operation::unwindows, {branch}, {}, side, shift);
                x              = append(file, Operation::add, {x, branch});
                branch         = append(file, Operation::norm, {x}, prefix + ".norm2");
                branch         = append(file, Operation::linear, {branch}, prefix + ".mlp.fc1");
                branch         = append(file, Operation::gelu, {branch});
                branch         = append(file, Operation::linear, {branch}, prefix + ".mlp.fc2");
                x              = append(file, Operation::add, {x, branch});
            }
            outputs[stage] = append(file, Operation::norm, {x}, std::format("bb.norm{}", stage));
            if (stage < 3) {
                x = append(file, Operation::merge, {x});
                x = append(file, Operation::norm, {x}, std::format("bb.layers.{}.downsample.norm", stage));
                x = append(file, Operation::linear, {x}, std::format("bb.layers.{}.downsample.reduction", stage));
            }
        }
        return outputs;
    }
    int Network::decoder_block(files::SafeFile& file, const int input, const std::string& name) {
        int x = append(file, Operation::convolution, {input}, name + ".conv_in");
        x     = append(file, Operation::batchnorm, {x}, name + ".bn_in");
        x     = append(file, Operation::relu, {x});
        std::vector<int> branches;
        for (int i = 0; i < 4; ++i) {
            const auto prefix = name + (i == 0 ? ".dec_att.aspp1" : std::format(".dec_att.aspp_deforms.{}", i - 1));
            constexpr std::array kernels{1, 1, 3, 7};
            const int offsets    = append(file, Operation::convolution, {x}, prefix + ".atrous_conv.offset_conv");
            const int modulation = append(file, Operation::convolution, {x}, prefix + ".atrous_conv.modulator_conv");
            int branch           = append(file, Operation::deform, {x, offsets, modulation}, {}, kernels[i]);
            branch               = append(file, Operation::linear, {branch}, prefix + ".atrous_conv.regular_conv");
            branch               = append(file, Operation::batchnorm, {branch}, prefix + ".bn");
            branches.push_back(append(file, Operation::relu, {branch}));
        }
        int pooled = append(file, Operation::pool, {x});
        pooled     = append(file, Operation::convolution, {pooled}, name + ".dec_att.global_avg_pool.1");
        pooled     = append(file, Operation::batchnorm, {pooled}, name + ".dec_att.global_avg_pool.2");
        pooled     = append(file, Operation::relu, {pooled});
        branches.push_back(append(file, Operation::resize, {pooled}, {}, tensors[x].h));
        x = append(file, Operation::join, std::move(branches));
        x = append(file, Operation::convolution, {x}, name + ".dec_att.conv1");
        x = append(file, Operation::batchnorm, {x}, name + ".dec_att.bn1");
        x = append(file, Operation::relu, {x});
        x = append(file, Operation::convolution, {x}, name + ".conv_out");
        return append(file, Operation::batchnorm, {x}, name + ".bn_out");
    }
    int Network::input_block(files::SafeFile& file, const int side, const int level) {
        int x = side == 1024 ? 0 : append(file, Operation::patches, {0}, {}, side);
        x     = append(file, Operation::convolution, {x}, std::format("decoder.ipt_blk{}.conv1", level));
        return append(file, Operation::convolution, {x}, std::format("decoder.ipt_blk{}.conv_out", level));
    }
    void Network::forward() {
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const auto& node  = nodes[i];
            const auto output = tensors[i + 1], input = tensors[node.inputs.front()];
            const auto& w = node.weights;
            switch (node.operation) {
            case Operation::convolution: runtime.convolution(output, input, {w[0], w[1], node.argument, node.second}); break;
            case Operation::linear: runtime.linear(output, input, {w[0], w[1]}); break;
            case Operation::norm: runtime.layer_norm(output, input, {w[0], w[1]}); break;
            case Operation::batchnorm: batch_norm(stream, output, input, w[0], w[1], w[2], w[3]); break;
            case Operation::gelu: compute::kernels::activation(stream, output.data, input.data, input.elements(), 2); break;
            case Operation::relu: relu(stream, output, input); break;
            case Operation::resize: resize(stream, output, input); break;
            case Operation::join:
                {
                    int offset{};
                    for (int source : node.inputs) {
                        join(stream, output, tensors[source], offset);
                        offset += tensors[source].c;
                    }
                    break;
                }
            case Operation::add:
            case Operation::gate: combine(stream, output, input, tensors[node.inputs[1]], node.operation == Operation::gate); break;
            case Operation::pool: pool(stream, output, input); break;
            case Operation::patches:
            case Operation::merge: patches(stream, output, input, node.operation == Operation::merge); break;
            case Operation::windows: windows(stream, output, input, node.argument, false); break;
            case Operation::unwindows: windows(stream, output, input, node.second, true); break;
            case Operation::bias: attention_bias(stream, output, w[0], node.argument, node.second != 0); break;
            case Operation::attention:
                {
                    auto query = input.reshape(input.n, 1, 144, output.c), key = query, value = query;
                    key.data   = static_cast<std::byte*>(input.data) + output.c * 2;
                    value.data = static_cast<std::byte*>(input.data) + output.c * 4;
                    runtime.attention(output, query, key, value, output.c / 32, input.c, input.c, false, nullptr, nullptr, tensors[node.inputs[1]]);
                    break;
                }
            case Operation::deform: deform_columns(stream, output, input, tensors[node.inputs[1]], tensors[node.inputs[2]], node.argument); break;
            }
        }
    }
} // namespace genesia::birefnet
