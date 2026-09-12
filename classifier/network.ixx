module;
#include "kernels.h"
#include <cublasLt.h>
#include <cudnn.h>

#include <nlohmann/json.hpp>
export module classifier.network;
export import classifier.storage;
import std;
export namespace classifier {
    void cuda_check(cudaError_t status);
    struct DeviceBuffer {
        void* data       = nullptr;
        std::size_t size = 0;
        bool owned       = true;
        DeviceBuffer()   = default;
        explicit DeviceBuffer(std::size_t bytes);
        DeviceBuffer(void* shared, std::size_t bytes);
        ~DeviceBuffer();
        DeviceBuffer(DeviceBuffer&& other) noexcept;
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
        DeviceBuffer(const DeviceBuffer&)            = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    };
    struct Parameter {
        std::string name;
        std::vector<std::int64_t> shape;
        DeviceBuffer storage;
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
        DeviceBuffer random_storage, update_storage;
        RandomState* random;
        UpdateState* update;
        explicit OptimizerState(const nlohmann::json& metadata);
    };
    enum class NetworkLoad { inference, resume, pretrained };
    struct Network {
        std::map<std::string, Parameter> parameters;
        std::vector<std::string> classes;
        nlohmann::json metadata;
        std::filesystem::path cache_directory;
        Affine stem, stem_norm;
        std::array<Stage, 4> stages;
        Head head;
        std::unique_ptr<OptimizerState> optimizer;
        explicit Network(const std::filesystem::path& path, NetworkLoad load = NetworkLoad::inference);
        void bind_layers();
        void initialize_head(const std::vector<std::string>& labels, std::uint64_t seed);
        void save(const std::filesystem::path& path, const nlohmann::json& state, bool include_optimizer = true);
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
    struct Gemm {
        cublasLtMatmulDesc_t descriptor = nullptr;
        cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
        cublasLtMatmulAlgo_t algorithm{};
        std::size_t workspace = 0;
        Gemm()                = default;
        ~Gemm();
        Gemm(const Gemm&) = delete;
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
        unsigned char* accepted;
    };
    enum class ActivationStorage { reuse, retain };
    struct NetworkPlan {
        Network& model;
        int batch, width, height;
        ActivationStorage storage;
        cudaStream_t stream   = nullptr;
        cublasLtHandle_t blas = nullptr;
        cudnnHandle_t cudnn   = nullptr;
        std::vector<Tensor> tensors;
        std::vector<Op> ops;
        DeviceBuffer arena, auxiliary, scratch, workspace, pixels;
        std::map<std::string, std::unique_ptr<Gemm>> gemms;
        std::map<std::string, ConvPlan> convolutions;
        cudaGraph_t inference_graph    = nullptr;
        cudaGraphExec_t inference_exec = nullptr;
        std::size_t allocated_bytes    = 0;
        NetworkPlan(Network& model, int batch, int width, int height, ActivationStorage storage = ActivationStorage::reuse, NetworkPlan* shared = nullptr);
        ~NetworkPlan();
        int append(Operation kind, std::string name, int input, int channels = 0, int kernel = 0, int skip = -1, float probability = 0, Affine affine = {});
        void matrix(std::string key, Tensor a, Tensor b, Tensor out, bool ta, bool tb, const void* bias = nullptr, float beta = 0);
        void convolution(std::string key, Tensor x, Parameter& weight, Tensor out, int kernel, int direction = 0, const void* original_input = nullptr);
        void forward(bool stochastic = false);
        void capture();
        void infer(GpuRgb8 input, GpuResult result, cudaStream_t caller_stream, float threshold = .9f);
    };
    struct TrainingPlan : NetworkPlan {
        int effective_batch;
        std::vector<Tensor> gradients;
        DeviceBuffer gradient_arena, labels, augment;
        cudaGraph_t training_graph = nullptr, frozen_graph = nullptr;
        cudaGraphExec_t training_exec = nullptr, frozen_exec = nullptr;
        TrainingPlan(Network& model, int batch, int width, int height, TrainingPlan* shared = nullptr);
        ~TrainingPlan();
        void backward(bool head_only);
        void capture_training();
        void microbatch(const unsigned char* host_rgb, const int* host_labels, bool head_only, cudaStream_t caller_stream, cudaEvent_t copied);
    };
} // namespace classifier
