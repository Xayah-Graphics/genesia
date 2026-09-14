module;
#include "../../compute/inference-kernels.h"
#include "../../compute/tensor.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>
module genesia.models.sdxl.lora;
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
} // namespace genesia::sdxl
