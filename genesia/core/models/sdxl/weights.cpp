module;
#include "../../compute/inference-kernels.h"
#include "../../compute/tensor.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>
module genesia.models.sdxl.weights;
import genesia.project;
import genesia.io.files;
import std;

namespace genesia::sdxl {
    Adapter::Adapter(const std::filesystem::path& path, const files::SafeFile& checkpoint) : file{path} {
        std::map<std::string, std::string> names;
        for (const auto& [name, entry] : checkpoint.header.items()) {
            if (!name.starts_with("model.diffusion_model.") || !name.ends_with(".weight") || entry.at("shape").size() < 2) continue;
            auto key = name.substr(22, name.size() - 22 - 7);
            std::ranges::replace(key, '.', '_');
            names.emplace("lora_unet_" + key, name);
        }
        std::set<std::string> groups;
        for (const auto& [name, entry] : file.header.items()) {
            if (name == "__metadata__") continue;
            std::string suffix;
            for (const std::string_view candidate : {".lora_down.weight", ".lora_up.weight", ".alpha"})
                if (name.ends_with(candidate)) suffix = candidate;
            if (suffix.empty() || !names.contains(name.substr(0, name.size() - suffix.size()))) throw std::runtime_error{"Unsupported SDXL UNet LoRA tensor: " + name};
            groups.insert(name.substr(0, name.size() - suffix.size()));
        }
        if (groups.empty()) throw std::runtime_error{"LoRA contains no UNet weights"};
        for (const auto& group : groups) {
            auto down          = tensor(group + ".lora_down.weight");
            auto up            = tensor(group + ".lora_up.weight");
            const auto alpha   = tensor(group + ".alpha");
            const auto& name   = names.at(group);
            const auto shape   = checkpoint.header.at(name).at("shape").get<std::vector<int>>();
            const int rank     = down.shape.at(0);
            auto expected_down = shape, expected_up = shape;
            expected_down[0] = rank;
            expected_up[1]   = rank;
            for (std::size_t i = 2; i < expected_up.size(); ++i) expected_up[i] = 1;
            if (rank <= 0 || (shape.size() != 2 && shape.size() != 4) || down.shape != expected_down || up.shape != expected_up || std::accumulate(alpha.shape.begin(), alpha.shape.end(), 1uz, std::multiplies<>{}) != 1) throw std::runtime_error{"Invalid LoRA dimensions: " + group};
            float scale{};
            if (alpha.scalar == compute::Scalar::f32) std::memcpy(&scale, alpha.data, sizeof(float));
            else if (alpha.scalar == compute::Scalar::f16) {
                __half value;
                std::memcpy(&value, alpha.data, sizeof(value));
                scale = __half2float(value);
            } else {
                __nv_bfloat16 value;
                std::memcpy(&value, alpha.data, sizeof(value));
                scale = __bfloat162float(value);
            }
            if (!std::isfinite(scale)) throw std::runtime_error{"Invalid LoRA alpha: " + group};
            layers.emplace(name, Layer{std::move(down), std::move(up), scale / rank});
        }
    }

    void Adapter::add(const Layer& layer, const float weight, const compute::TensorView output, compute::InferenceRuntime& runtime) const {
        const auto stream = runtime.stream;
        const auto pool   = ::cuda::device_default_memory_pool(stream.device());
        std::array<::cuda::device_buffer<float>, 2> values{::cuda::device_buffer<float>{stream, pool}, ::cuda::device_buffer<float>{stream, pool}};
        int index{};
        for (const auto* tensor : {&layer.down, &layer.up}) {
            const auto count = std::accumulate(tensor->shape.begin(), tensor->shape.end(), 1uz, std::multiplies<>{});
            values[index]    = ::cuda::device_buffer<float>{stream, pool, count, ::cuda::no_init};
            ::cuda::device_buffer<std::byte> source{stream, pool, tensor->bytes, ::cuda::no_init};
            ::cuda::copy_bytes(stream, ::cuda::std::span<const std::byte>{tensor->data, tensor->bytes}, source);
            compute::kernels::convert(stream, values[index++].data(), source.data(), count, int(tensor->scalar), int(compute::Scalar::f32));
        }
        const int rank = layer.down.shape[0];
        runtime.add_product(output, {values[1].data(), 1, 1, output.w, rank, compute::Scalar::f32}, {values[0].data(), 1, 1, rank, output.c, compute::Scalar::f32}, weight * layer.scale);
    }

    Adapter::Tensor Adapter::tensor(const std::string& name) const {
        const auto& entry  = file.header.at(name);
        const auto offsets = entry.at("data_offsets").get<std::array<std::size_t, 2>>();
        static const std::map<std::string, compute::Scalar> types{{"F16", compute::Scalar::f16}, {"F32", compute::Scalar::f32}, {"BF16", compute::Scalar::bf16}};
        return {file.base + offsets[0], offsets[1] - offsets[0], entry.at("shape").get<std::vector<int>>(), types.at(entry.at("dtype").get<std::string>())};
    }

    void Weights::apply(const std::filesystem::path& checkpoint, const std::span<const generation::Lora> loras, compute::InferenceRuntime& runtime) {
        if (std::ranges::equal(applied, loras)) return;
        files::SafeFile base{checkpoint};
        std::vector<std::unique_ptr<Adapter>> adapters;
        std::map<std::string, Target> changed;
        for (const auto& name : patched) changed.emplace(name, targets.at(name));
        for (const auto& selection : loras) {
            auto adapter = std::make_unique<Adapter>(std::filesystem::path{project::assets} / "loras" / files::path(selection.file), base);
            for (const auto& [name, layer] : adapter->layers) changed.emplace(name, targets.at(name));
            adapters.push_back(std::move(adapter));
        }
        merge(base, loras, adapters, changed, runtime);
        patched.clear();
        for (const auto& adapter : adapters)
            for (const auto& [name, layer] : adapter->layers) patched.insert(name);
        applied.assign(loras.begin(), loras.end());
    }

    Weights::Variant Weights::variant(const std::filesystem::path& checkpoint, const std::span<const generation::Lora> loras, const std::span<const generation::Lora> added, const Variant& previous, compute::InferenceRuntime& runtime) {
        files::SafeFile base{checkpoint};
        std::vector<std::unique_ptr<Adapter>> adapters;
        std::map<std::string, Target> changed;
        std::set<std::size_t> blocks;
        for (const auto& selection : loras) {
            auto adapter = std::make_unique<Adapter>(std::filesystem::path{project::assets} / "loras" / files::path(selection.file), base);
            if (std::ranges::contains(added, selection.file, &generation::Lora::file))
                for (const auto& [name, layer] : adapter->layers) {
                    const auto& target = targets.at(name);
                    changed.emplace(name, target);
                    blocks.insert(target.block);
                }
            adapters.push_back(std::move(adapter));
        }
        Variant result;
        result.bindings = previous.bindings;
        // Fused QKV matrices share an allocation; preserve its untouched slices.
        for (const auto block : blocks) {
            const auto& original = storage[block];
            auto& copy           = result.storage.emplace_back(runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), original.size(), ::cuda::no_init);
            const auto parent    = previous.bindings.find(original.data());
            const auto* source   = parent == previous.bindings.end() ? original.data() : static_cast<const std::byte*>(parent->second);
            ::cuda::copy_bytes(runtime.stream, ::cuda::std::span<const std::byte>{source, original.size()}, copy);
            result.bindings[original.data()] = copy.data();
        }
        for (auto& [name, target] : changed) {
            const auto* original = storage[target.block].data();
            const auto offset    = static_cast<const std::byte*>(target.view.data) - original;
            target.view.data     = static_cast<std::byte*>(result.bindings.at(original)) + offset;
        }
        merge(base, loras, adapters, changed, runtime);
        return result;
    }

    void Weights::merge(const files::SafeFile& base, const std::span<const generation::Lora> loras, const std::span<const std::unique_ptr<Adapter>> adapters, const std::map<std::string, Target>& destinations, compute::InferenceRuntime& runtime) {
        runtime.begin_preparation();
        const auto stream = runtime.stream;
        const auto pool   = ::cuda::device_default_memory_pool(stream.device());
        static const std::map<std::string, compute::Scalar> types{{"F16", compute::Scalar::f16}, {"F32", compute::Scalar::f32}, {"BF16", compute::Scalar::bf16}};
        try {
            for (const auto& [name, target] : destinations) {
                const auto& entry  = base.header.at(name);
                const auto shape   = entry.at("shape").get<std::vector<int>>();
                const auto offsets = entry.at("data_offsets").get<std::array<std::size_t, 2>>();
                const auto count   = target.view.elements();
                ::cuda::device_buffer<std::byte> source{stream, pool, offsets[1] - offsets[0], ::cuda::no_init};
                ::cuda::device_buffer<float> combined{stream, pool, count, ::cuda::no_init};
                ::cuda::copy_bytes(stream, ::cuda::std::span<const std::byte>{base.base + offsets[0], source.size()}, source);
                compute::kernels::convert(stream, combined.data(), source.data(), count, int(types.at(entry.at("dtype").get<std::string>())), int(compute::Scalar::f32));
                const compute::TensorView matrix{combined.data(), 1, 1, shape[0], int(count / shape[0]), compute::Scalar::f32};
                for (std::size_t i = 0; i < adapters.size(); ++i) {
                    const auto layer = adapters[i]->layers.find(name);
                    if (layer != adapters[i]->layers.end()) adapters[i]->add(layer->second, loras[i].weight, matrix, runtime);
                }
                if (target.convolution) compute::kernels::convert_layout(stream, target.view.data, combined.data(), target.view.n, target.view.h * target.view.w, target.view.c, int(compute::Scalar::f32), int(target.view.scalar));
                else compute::kernels::convert(stream, target.view.data, combined.data(), count, int(compute::Scalar::f32), int(target.view.scalar));
            }
        } catch (...) {
            stream.sync();
            throw;
        }
        stream.sync();
    }

    Checkpoint::Checkpoint(const std::filesystem::path& path, compute::InferenceRuntime& execution, Weights& storage) : runtime{execution}, file{path}, weights{storage} {}
    Checkpoint::~Checkpoint() {
        runtime.stream.sync();
    }

    compute::TensorView Checkpoint::tensor(const std::string& name, const compute::Scalar scalar, const bool convolution, const bool transpose) {
        const auto shape = file.header.at(name).at("shape").get<std::vector<int>>();
        const auto count = std::accumulate(shape.begin(), shape.end(), 1uz, std::multiplies<>{});
        auto& storage    = weights.storage.emplace_back(runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), count * (scalar == compute::Scalar::f32 ? 4uz : 2uz), ::cuda::no_init);
        compute::TensorView result{storage.data(), 1, 1, shape.size() >= 2 ? shape[0] : 1, shape.size() >= 2 ? shape[1] : shape[0], scalar};
        if (convolution) result = {storage.data(), shape[0], shape[2], shape[3], shape[1], scalar};
        else if (transpose) std::swap(result.w, result.c);
        load(name, result, convolution, transpose);
        return result;
    }

    compute::Linear Checkpoint::linear(const std::string& prefix, const compute::Scalar scalar, const bool bias) {
        return {tensor(prefix + ".weight", scalar), bias ? tensor(prefix + ".bias", scalar) : compute::TensorView{}};
    }

    compute::Linear Checkpoint::qkv(const std::span<const std::string> prefixes, const compute::Scalar scalar, const bool bias) {
        compute::Linear result;
        for (const bool is_bias : {false, true}) {
            if (is_bias && !bias) continue;
            const std::string suffix = is_bias ? ".bias" : ".weight";
            const auto shape         = file.header.at(prefixes[0] + suffix).at("shape").get<std::vector<int>>();
            const auto count         = std::accumulate(shape.begin(), shape.end(), 1uz, std::multiplies<>{});
            const auto bytes         = count * (scalar == compute::Scalar::f32 ? 4uz : 2uz);
            auto& storage            = weights.storage.emplace_back(runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), bytes * prefixes.size(), ::cuda::no_init);
            for (std::size_t i = 0; i < prefixes.size(); ++i) load(prefixes[i] + suffix, {storage.data() + i * bytes, 1, 1, is_bias ? 1 : shape[0], is_bias ? shape[0] : shape[1], scalar});
            if (is_bias) result.bias = {storage.data(), 1, 1, 1, shape[0] * int(prefixes.size()), scalar};
            else result.weight = {storage.data(), 1, 1, shape[0] * int(prefixes.size()), shape[1], scalar};
        }
        return result;
    }

    compute::Norm Checkpoint::norm(const std::string& prefix, const compute::Scalar scalar, const float epsilon) {
        return {tensor(prefix + ".weight", scalar), tensor(prefix + ".bias", scalar), epsilon};
    }
    compute::Conv Checkpoint::convolution(const std::string& prefix, const compute::Scalar scalar, const int stride, const int padding) {
        return {tensor(prefix + ".weight", scalar, true), tensor(prefix + ".bias", scalar), stride, padding};
    }

    void Checkpoint::load(const std::string& name, const compute::TensorView destination, const bool convolution, const bool transpose) {
        const auto& entry  = file.header.at(name);
        const auto offsets = entry.at("data_offsets").get<std::array<std::size_t, 2>>();
        static const std::map<std::string, compute::Scalar> types{{"F16", compute::Scalar::f16}, {"F32", compute::Scalar::f32}, {"BF16", compute::Scalar::bf16}};
        const auto scalar = types.at(entry.at("dtype").get<std::string>());
        if (!convolution && !transpose && scalar == destination.scalar) ::cuda::copy_bytes(runtime.stream, ::cuda::std::span<const std::byte>{file.base + offsets[0], destination.bytes()}, ::cuda::std::span<std::byte>{static_cast<std::byte*>(destination.data), destination.bytes()});
        else {
            ::cuda::device_buffer<std::byte> source{runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), offsets[1] - offsets[0], ::cuda::no_init};
            ::cuda::copy_bytes(runtime.stream, ::cuda::std::span<const std::byte>{file.base + offsets[0], source.size()}, source);
            if (convolution) compute::kernels::convert_layout(runtime.stream, destination.data, source.data(), destination.n, destination.h * destination.w, destination.c, int(scalar), int(destination.scalar));
            else if (transpose) compute::kernels::convert_layout(runtime.stream, destination.data, source.data(), 1, destination.w, destination.c, int(scalar), int(destination.scalar));
            else compute::kernels::convert(runtime.stream, destination.data, source.data(), destination.elements(), int(scalar), int(destination.scalar));
        }
        if (name.starts_with("model.diffusion_model.") && name.ends_with(".weight") && entry.at("shape").size() >= 2) weights.targets.emplace(name, Weights::Target{destination, weights.storage.size() - 1, convolution});
    }
} // namespace genesia::sdxl
