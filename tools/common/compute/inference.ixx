module;

#include "tensor.h"
#include <cublasLt.h>
#include <cudnn.h>
#include <genesia/cuda.h>

export module tools.compute.inference;

export import tools.compute.plans;
import std;

export namespace tools::compute {
    struct Linear final {
        TensorView weight;
        TensorView bias;
    };
    struct Norm final {
        TensorView weight;
        TensorView bias;
        float epsilon{1.0e-5F};
    };
    struct Conv final {
        TensorView weight;
        TensorView bias;
        int stride{1};
        int padding{1};
    };

    struct MatmulShape final {
        int rows;
        int columns;
        int reduction;
        bool bias;

        bool operator==(const MatmulShape&) const = default;
    };

    struct MatmulPlan final : MatrixPlan {
        MatmulShape shape;
        explicit MatmulPlan(const MatmulShape& shape);
        MatmulPlan(const MatmulPlan&)            = delete;
        MatmulPlan& operator=(const MatmulPlan&) = delete;
    };

    struct InferenceRuntime final {
        ::cuda::stream_ref stream;

    private:
        struct ConvPlan;
        struct AttentionPlan;

        Handles handles;
        ::cuda::device_buffer<std::byte> workspace;
        std::filesystem::path cache_directory;
        void* workspace_data{};
        std::size_t workspace_bytes{};
        std::size_t required_workspace{};
        std::list<MatmulPlan> matmuls;
        std::list<ConvPlan> convolutions;
        std::list<AttentionPlan> attentions;

    public:
        InferenceRuntime(::cuda::stream_ref stream, const std::filesystem::path& cache_directory);
        ~InferenceRuntime();
        InferenceRuntime(const InferenceRuntime&)            = delete;
        InferenceRuntime& operator=(const InferenceRuntime&) = delete;
        ::cuda::device_buffer<std::byte> finish_preparation();
        void linear(TensorView output, TensorView input, const Linear& layer);
        void convolution(TensorView output, TensorView input, const Conv& layer);
        void layer_norm(TensorView output, TensorView input, const Norm& layer);
        void attention(TensorView output, TensorView query, TensorView key, TensorView value, int heads, int query_stride, int key_stride, TensorView bias);
    };

} // namespace tools::compute
