module;
#include "../../compute/inference-kernels.h"
#include "../../compute/tensor.h"
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>
module genesia.models.sdxl.weights;
import genesia.project;
import std;

namespace genesia::sdxl {
    void Weights::apply(const std::filesystem::path& checkpoint, const std::span<const generation::Lora> loras, compute::InferenceRuntime& runtime) {
        if (std::ranges::equal(applied, loras)) return;
        files::SafeFile base{checkpoint};
        std::vector<std::unique_ptr<Adapter>> adapters;
        std::map<std::string, Target> changed;
        for (const auto& name : patched) changed.emplace(name, targets.at(name));
        for (const auto& selection : loras) {
            auto adapter = std::make_unique<Adapter>(project::directory / files::path(selection.concept_key) / ".genesia" / "models" / (selection.sha + ".safetensors"), base);
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
            auto adapter = std::make_unique<Adapter>(project::directory / files::path(selection.concept_key) / ".genesia" / "models" / (selection.sha + ".safetensors"), base);
            if (std::ranges::contains(added, selection.concept_key, &generation::Lora::concept_key))
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
