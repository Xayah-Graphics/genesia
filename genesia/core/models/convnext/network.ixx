module;
#include "../../compute/tensor.h"
#include "kernels.h"
#include <cublasLt.h>
#include <cudnn.h>
export module genesia.models.convnext;
export import genesia.io.safetensors;
export import genesia.compute.plans;
import std;
export namespace genesia::convnext {
    struct Parameter {
        std::string name;
        std::vector<std::int64_t> shape;
        compute::DeviceBuffer storage;
        ParameterView gpu{};
    };
    struct Affine {
        Parameter* weight = nullptr;
        Parameter* bias   = nullptr;
    };
    struct Block {
        Affine depthwise, norm, expand, grn, project;
        float drop_path;
    };
    struct Stage {
        Affine norm, downsample;
        std::vector<Block> blocks;
    };
    struct Head {
        Affine norm, linear;
    };
    struct OptimizerState {
        compute::DeviceBuffer random_storage, update_storage;
        RandomState* random;
        UpdateState* update;
        OptimizerState();
    };
    enum class NetworkLoad { inference, resume, pretrained };
    struct Network {
        std::map<std::string, Parameter> parameters;
        std::vector<std::string> classes;
        int step{};
        std::uint64_t sequence{};
        std::filesystem::path cache_directory;
        Affine stem, stem_norm;
        std::array<Stage, 4> stages;
        Head head;
        std::unique_ptr<OptimizerState> optimizer;
        explicit Network(const std::filesystem::path& path, NetworkLoad load = NetworkLoad::inference);
        void bind_layers();
        void initialize_head(const std::vector<std::string>& labels, std::uint64_t seed);
        void save(const std::filesystem::path& path, std::optional<std::string_view> training_state = {});
    };
    enum class Operation { convolution, depthwise, norm, linear, gelu, grn, residual, pool, dropout };
    struct Op {
        Operation kind;
        std::string name;
        int input = 0, skip = -1, output = 0, kernel = 0;
        Parameter* weight     = nullptr;
        Parameter* bias       = nullptr;
        float probability     = 0;
        float* aux            = nullptr;
        std::size_t aux_bytes = 0;
    };
    struct ConvPlan {
        std::shared_ptr<cudnn_frontend::graph::Graph> graph;
        std::size_t workspace = 0;
    };
    struct GpuRgb8 {
        const unsigned char* pixels;
        int width, height;
        std::size_t row_stride;
    };
    struct GpuResult {
        float* scores;
        int* indices;
    };
    enum class ActivationStorage { reuse, retain };
    struct NetworkPlan {
        Network& model;
        int batch, width, height;
        ActivationStorage storage;
        cudaStream_t stream = nullptr;
        std::optional<compute::Handles> handles;
        std::vector<compute::TensorView> tensors;
        std::vector<Op> ops;
        compute::DeviceBuffer arena, auxiliary, scratch, workspace, pixels;
        std::map<std::string, std::unique_ptr<compute::MatrixPlan>> gemms;
        std::map<std::string, ConvPlan> convolutions;
        cudaGraph_t inference_graph    = nullptr;
        cudaGraphExec_t inference_exec = nullptr;
        std::size_t allocated_bytes    = 0;
        NetworkPlan(Network& model, int batch, int width, int height, ActivationStorage storage = ActivationStorage::reuse, NetworkPlan* shared = nullptr);
        ~NetworkPlan();
        int append(Operation kind, std::string name, int input, int channels = 0, int kernel = 0, int skip = -1, float probability = 0, Affine affine = {});
        void matrix(std::string key, compute::TensorView a, compute::TensorView b, compute::TensorView out, bool ta, bool tb, const void* bias = nullptr, float beta = 0);
        void convolution(std::string key, compute::TensorView x, Parameter& weight, compute::TensorView out, int kernel, int direction = 0, const void* original_input = nullptr);
        void forward(bool stochastic = false);
        void capture();
        void infer(GpuRgb8 input, GpuResult result, cudaStream_t caller_stream);
    };
    struct TrainingPlan : NetworkPlan {
        int effective_batch;
        std::vector<compute::TensorView> gradients;
        compute::DeviceBuffer gradient_arena, labels, augment;
        cudaGraph_t training_graph = nullptr, frozen_graph = nullptr;
        cudaGraphExec_t training_exec = nullptr, frozen_exec = nullptr;
        TrainingPlan(Network& model, int batch, int width, int height, TrainingPlan* shared = nullptr);
        ~TrainingPlan();
        void backward(bool head_only);
        void capture_training();
        void microbatch(const unsigned char* host_rgb, const int* host_labels, bool head_only, cudaStream_t caller_stream, cudaEvent_t copied);
    };
} // namespace genesia::convnext
