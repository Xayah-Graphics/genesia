module;
#include "../../compute/tensor.h"
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>

export module genesia.models.sdxl.weights;
import genesia.io.safetensors;
import std;
import genesia.compute.inference;
import genesia.generation.settings;

export namespace genesia::sdxl {
    struct Adapter final {
        struct Tensor final {
            const std::byte* data;
            std::size_t bytes;
            std::vector<int> shape;
            compute::Scalar scalar;
        };
        struct Layer final {
            Tensor down, up;
            float scale;
        };
        files::SafeFile file;
        std::map<std::string, Layer> layers;

        Adapter(const std::filesystem::path& path, const files::SafeFile& checkpoint);
        void add(const Layer& layer, float weight, compute::TensorView output, compute::InferenceRuntime& runtime) const;

    private:
        Tensor tensor(const std::string& name) const;
    };

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
