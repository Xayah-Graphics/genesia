module;
#include "kernels.h"
#include <genesia/cuda.h>
export module foreground.models.birefnet;
import tools.safetensors;
import tools.compute.inference;
import std;
export namespace foreground::birefnet {
    struct Network final {
        explicit Network(const std::filesystem::path& model);
        ~Network();
        std::vector<std::uint8_t> infer(std::span<const std::uint8_t> rgb);

    private:
        enum class Operation { convolution, linear, norm, batchnorm, gelu, relu, resize, join, add, gate, pool, patches, merge, windows, unwindows, bias, attention, deform };
        struct Node final {
            Operation operation;
            std::vector<int> inputs;
            std::array<tools::compute::TensorView, 4> weights{};
            int argument{}, second{};
        };
        ::cuda::stream stream;
        tools::compute::InferenceRuntime runtime;
        std::vector<tools::compute::DeviceBuffer> weights;
        std::map<std::string, tools::compute::TensorView> parameters;
        std::vector<tools::compute::TensorView> tensors;
        std::vector<Node> nodes;
        tools::compute::DeviceBuffer arena, pixels, mask;
        ::cuda::device_buffer<std::byte> operator_workspace;
        std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cudaGraphDestroy)> graph{nullptr, cudaGraphDestroy};
        std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, decltype(&cudaGraphExecDestroy)> executable{nullptr, cudaGraphExecDestroy};
        tools::compute::TensorView parameter(tools::files::SafeFile& file, const std::string& name);
        int append(tools::files::SafeFile& file, Operation operation, std::vector<int> inputs, std::string name = {}, int argument = 0, int second = 0);
        std::array<int, 4> backbone(tools::files::SafeFile& file, int input);
        int decoder_block(tools::files::SafeFile& file, int input, const std::string& name);
        int input_block(tools::files::SafeFile& file, int side, int level);
        void forward();
    };
} // namespace foreground::birefnet
