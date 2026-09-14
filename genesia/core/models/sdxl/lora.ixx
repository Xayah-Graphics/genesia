module;
#include "../../compute/tensor.h"
#include <genesia/cuda.h>
export module genesia.models.sdxl.lora;
export import genesia.io.safetensors;
import genesia.compute.inference;
import std;

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
} // namespace genesia::sdxl
