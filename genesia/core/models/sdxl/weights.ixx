module;
#include "../../compute/tensor.h"
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>

export module genesia.models.sdxl.weights;
import genesia.io.safetensors;
import std;
import genesia.compute.inference;

export namespace genesia::sdxl {
    struct Weights final {
        std::deque<::cuda::device_buffer<std::byte>> storage;
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
    };
} // namespace genesia::sdxl
