module;

#include <cublasLt.h>
#include <cudnn.h>
#include <genesia/cuda.h>

export module genesia.neural.inference_runtime;

import std;

export namespace genesia::neural {
    enum class Scalar : std::uint8_t { f32, f16, bf16 };

    struct TensorView final {
        void* data{};
        int n{1};
        int h{1};
        int w{1};
        int c{1};
        Scalar scalar{Scalar::f16};

        std::size_t elements() const;
        std::size_t bytes() const;
        TensorView reshape(int n, int h, int w, int c, Scalar scalar = Scalar::f16) const;
    };

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

    struct MatmulPlan final {
        MatmulShape shape;
        std::unique_ptr<std::remove_pointer_t<cublasLtMatmulDesc_t>, decltype(&cublasLtMatmulDescDestroy)> operation{nullptr, cublasLtMatmulDescDestroy};
        std::unique_ptr<std::remove_pointer_t<cublasLtMatrixLayout_t>, decltype(&cublasLtMatrixLayoutDestroy)> a{nullptr, cublasLtMatrixLayoutDestroy};
        std::unique_ptr<std::remove_pointer_t<cublasLtMatrixLayout_t>, decltype(&cublasLtMatrixLayoutDestroy)> b{nullptr, cublasLtMatrixLayoutDestroy};
        std::unique_ptr<std::remove_pointer_t<cublasLtMatrixLayout_t>, decltype(&cublasLtMatrixLayoutDestroy)> output{nullptr, cublasLtMatrixLayoutDestroy};
        cublasLtMatmulAlgo_t algorithm{};
        std::size_t workspace_bytes{};

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

        std::unique_ptr<std::remove_pointer_t<cublasLtHandle_t>, decltype(&cublasLtDestroy)> blas{nullptr, cublasLtDestroy};
        std::unique_ptr<std::remove_pointer_t<cudnnHandle_t>, decltype(&cudnnDestroy)> dnn{nullptr, cudnnDestroy};
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
        void geglu(TensorView output, TensorView input, const Linear& layer);
        void convolution(TensorView output, TensorView input, const Conv& layer, TensorView residual = {});
        void layer_norm(TensorView output, TensorView input, const Norm& layer);
        void group_norm(TensorView output, TensorView input, const Norm& layer, float* statistics, bool silu, bool prepared = false, TensorView time = {}, const int* step = nullptr);
        void attention(TensorView output, TensorView query, TensorView key, TensorView value, int heads, int query_stride, int key_stride, bool causal = false, const std::int32_t* positions = nullptr, const std::int32_t* lengths = nullptr);

    private:
        void* scratch(std::size_t bytes);
        void matmul(TensorView output, TensorView a, TensorView b, const MatmulShape& shape, float alpha = 1.0F, TensorView bias = {}, TensorView residual = {});
    };

    void check(cudaError_t status);
    void check(cublasStatus_t status);
    void check(cudnnStatus_t status);
} // namespace genesia::neural
