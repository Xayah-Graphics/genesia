module;
#include "../../compute/tensor.h"
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>

export module genesia.models.sdxl.weights;
import genesia.io.safetensors;
import genesia.models.sdxl.lora;
import std;
import genesia.compute.inference;
import genesia.generation.settings;

export namespace genesia::sdxl {
    struct Weights final {
        std::deque<::cuda::device_buffer<std::byte>> storage;
        struct Target final {
            compute::TensorView view;
            std::size_t block;
            bool convolution{};
        };
        std::map<std::string, Target> targets;
        std::vector<generation::Lora> applied;
        struct Variant final {
            std::vector<::cuda::device_buffer<std::byte>> storage;
            std::map<const void*, void*> bindings;
        };
        void apply(const std::filesystem::path& checkpoint, std::span<const generation::Lora> loras, compute::InferenceRuntime& runtime);
        Variant variant(const std::filesystem::path& checkpoint, std::span<const generation::Lora> loras, std::span<const generation::Lora> added, const Variant& previous, compute::InferenceRuntime& runtime);

    private:
        std::set<std::string> patched;
        void merge(const files::SafeFile& base, std::span<const generation::Lora> loras, std::span<const std::unique_ptr<Adapter>> adapters, const std::map<std::string, Target>& destinations, compute::InferenceRuntime& runtime);
    };

    struct Checkpoint final {
        compute::InferenceRuntime& runtime;

        Checkpoint(const std::filesystem::path& path, compute::InferenceRuntime& runtime, Weights& weights);
        ~Checkpoint();
        Checkpoint(const Checkpoint&)            = delete;
        Checkpoint& operator=(const Checkpoint&) = delete;
        compute::TensorView tensor(const std::string& name, compute::Scalar scalar, bool convolution = false, bool transpose = false);
        compute::Linear linear(const std::string& prefix, compute::Scalar scalar, bool bias = true);
        compute::Linear qkv(std::span<const std::string> prefixes, compute::Scalar scalar, bool bias);
        compute::Norm norm(const std::string& prefix, compute::Scalar scalar, float epsilon = 1.0e-5F);
        compute::Conv convolution(const std::string& prefix, compute::Scalar scalar, int stride = 1, int padding = 1);

    private:
        files::SafeFile file;
        Weights& weights;
        void load(const std::string& name, compute::TensorView destination, bool convolution = false, bool transpose = false);
    };
} // namespace genesia::sdxl
