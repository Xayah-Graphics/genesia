module;

#include "tensor.h"
#include <cublasLt.h>
#include <cudnn.h>
#include <genesia/cuda.h>

export module genesia.compute.inference;

export import genesia.compute.plans;
import std;

export namespace genesia::compute {
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
        int leading_a;
        int leading_b;
        Scalar scalar;
        Scalar result_scalar;
        bool transpose_a;
        bool bias;
        bool residual;

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
        std::size_t cache_hits{};
        std::size_t cache_misses{};

    private:
        struct ConvPlan;
        struct AttentionPlan;

        Handles handles;
        ::cuda::device_buffer<std::byte> workspace;
        ::cuda::device_buffer<std::byte> intermediate;
        std::filesystem::path cache_directory;
        void* workspace_data{};
        std::size_t workspace_bytes{};
        std::size_t required_workspace{};
        void* intermediate_data{};
        std::size_t intermediate_bytes{};
        std::list<MatmulPlan> matmuls;
        std::list<ConvPlan> convolutions;
        std::list<AttentionPlan> attentions;

    public:
        InferenceRuntime(::cuda::stream_ref stream, const std::filesystem::path& cache_directory);
        ~InferenceRuntime();
        InferenceRuntime(const InferenceRuntime&)            = delete;
        InferenceRuntime& operator=(const InferenceRuntime&) = delete;
        void begin_preparation();
        ::cuda::device_buffer<std::byte> finish_preparation();
        void linear(TensorView output, TensorView input, const Linear& layer, TensorView residual = {});
        void add_product(TensorView output, TensorView up, TensorView down, float scale);
        void geglu(TensorView output, TensorView input, const Linear& layer);
        void convolution(TensorView output, TensorView input, const Conv& layer, TensorView residual = {});
        void layer_norm(TensorView output, TensorView input, const Norm& layer);
        void group_norm(TensorView output, TensorView input, const Norm& layer, float* statistics, bool silu, bool prepared = false, TensorView time = {}, const int* step = nullptr);
        void attention(TensorView output, TensorView query, TensorView key, TensorView value, int heads, int query_stride, int key_stride, bool causal = false, const std::int32_t* positions = nullptr, const std::int32_t* lengths = nullptr);

    private:
        void* scratch(std::size_t bytes);
        void matmul(TensorView output, TensorView a, TensorView b, const MatmulShape& shape, float alpha = 1.0F, TensorView bias = {}, TensorView residual = {});
    };

} // namespace genesia::compute
