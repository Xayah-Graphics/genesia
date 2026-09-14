module;
#include "../../compute/inference-kernels.h"
#include "../../compute/tensor.h"
#include "kernels.h"
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>

module genesia.models.sdxl.weights;
import std;
import genesia.compute.inference;

namespace genesia::sdxl {
    Checkpoint::Checkpoint(const std::filesystem::path& path, compute::InferenceRuntime& execution, Weights& storage) : runtime{execution}, file{path}, weights{storage} {}

    Checkpoint::~Checkpoint() {
        runtime.stream.sync();
    }

    compute::TensorView Checkpoint::tensor(const std::string& name, const compute::Scalar scalar, const bool convolution, const bool transpose) {
        const auto& entry       = file.header.at(name);
        const auto shape        = entry.at("shape").get<std::vector<int>>();
        const auto offsets      = entry.at("data_offsets").get<std::array<std::size_t, 2>>();
        const std::size_t count = std::accumulate(shape.begin(), shape.end(), 1uz, std::multiplies<>{});
        static const std::unordered_map<std::string, compute::Scalar> types{{"F16", compute::Scalar::f16}, {"F32", compute::Scalar::f32}, {"BF16", compute::Scalar::bf16}};
        const compute::Scalar source_scalar = types.at(entry.at("dtype").get<std::string>());
        auto& destination                   = weights.storage.emplace_back(runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), count * (scalar == compute::Scalar::f32 ? 4uz : 2uz), ::cuda::no_init);
        compute::TensorView result{destination.data(), 1, 1, shape.size() >= 2 ? shape[0] : 1, shape.size() >= 2 ? shape[1] : shape[0], scalar};
        if (!convolution && !transpose && source_scalar == scalar) {
            ::cuda::copy_bytes(runtime.stream, ::cuda::std::span<const std::byte>{file.base + offsets[0], destination.size()}, destination);
        } else {
            ::cuda::device_buffer<std::byte> source{runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), offsets[1] - offsets[0], ::cuda::no_init};
            ::cuda::copy_bytes(runtime.stream, ::cuda::std::span<const std::byte>{file.base + offsets[0], source.size()}, source);
            if (convolution) {
                result = compute::TensorView{destination.data(), shape[0], shape[2], shape[3], shape[1], scalar};
                compute::kernels::convert_layout(runtime.stream, result.data, source.data(), result.n, result.h * result.w, result.c, int(source_scalar), int(scalar));
            } else if (transpose) {
                compute::kernels::convert_layout(runtime.stream, result.data, source.data(), 1, shape[1], shape[0], int(source_scalar), int(scalar));
                result.w = shape[1];
                result.c = shape[0];
            } else compute::kernels::convert(runtime.stream, result.data, source.data(), count, int(source_scalar), int(scalar));
        }
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
            const std::size_t count  = std::accumulate(shape.begin(), shape.end(), 1uz, std::multiplies<>{});
            auto& destination        = weights.storage.emplace_back(runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), count * prefixes.size() * (scalar == compute::Scalar::f32 ? 4uz : 2uz), ::cuda::no_init);
            auto packed              = ::cuda::device_buffer<std::byte>{runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), scalar == compute::Scalar::f16 ? 0uz : count * prefixes.size() * 2, ::cuda::no_init};
            std::byte* target        = scalar == compute::Scalar::f16 ? destination.data() : packed.data();
            for (int i = 0; i < prefixes.size(); ++i) {
                const auto offset = file.header.at(prefixes[i] + suffix).at("data_offsets")[0].get<std::size_t>();
                ::cuda::copy_bytes(runtime.stream, ::cuda::std::span<const std::byte>{file.base + offset, count * 2}, ::cuda::std::span<std::byte>{target + i * count * 2, count * 2});
            }
            if (scalar != compute::Scalar::f16) compute::kernels::convert(runtime.stream, destination.data(), packed.data(), count * prefixes.size(), 1, static_cast<int>(scalar));
            if (is_bias) result.bias = compute::TensorView{destination.data(), 1, 1, 1, shape[0] * static_cast<int>(prefixes.size()), scalar};
            else result.weight = compute::TensorView{destination.data(), 1, 1, shape[0] * static_cast<int>(prefixes.size()), shape[1], scalar};
        }
        return result;
    }

    compute::Norm Checkpoint::norm(const std::string& prefix, const compute::Scalar scalar, const float epsilon) {
        return {tensor(prefix + ".weight", scalar), tensor(prefix + ".bias", scalar), epsilon};
    }
    compute::Conv Checkpoint::convolution(const std::string& prefix, const compute::Scalar scalar, const int stride, const int padding) {
        return {tensor(prefix + ".weight", scalar, true), tensor(prefix + ".bias", scalar), stride, padding};
    }
} // namespace genesia::sdxl
