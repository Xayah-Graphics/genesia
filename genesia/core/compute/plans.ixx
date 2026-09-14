module;
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cudnn.h>
export module genesia.compute.plans;
export import genesia.compute.device;
import std;
export namespace genesia::compute {
    struct Handles final {
        std::unique_ptr<std::remove_pointer_t<cublasLtHandle_t>, decltype(&cublasLtDestroy)> blas{nullptr, cublasLtDestroy};
        std::unique_ptr<std::remove_pointer_t<cudnnHandle_t>, decltype(&cudnnDestroy)> dnn{nullptr, cudnnDestroy};
        explicit Handles(cudaStream_t stream);
    };
    struct MatrixPlan {
        std::unique_ptr<std::remove_pointer_t<cublasLtMatmulDesc_t>, decltype(&cublasLtMatmulDescDestroy)> operation{nullptr, cublasLtMatmulDescDestroy};
        std::unique_ptr<std::remove_pointer_t<cublasLtMatrixLayout_t>, decltype(&cublasLtMatrixLayoutDestroy)> a{nullptr, cublasLtMatrixLayoutDestroy};
        std::unique_ptr<std::remove_pointer_t<cublasLtMatrixLayout_t>, decltype(&cublasLtMatrixLayoutDestroy)> b{nullptr, cublasLtMatrixLayoutDestroy};
        std::unique_ptr<std::remove_pointer_t<cublasLtMatrixLayout_t>, decltype(&cublasLtMatrixLayoutDestroy)> output{nullptr, cublasLtMatrixLayoutDestroy};
        cublasLtMatmulAlgo_t algorithm{};
        std::size_t workspace_bytes{};
    };
    std::filesystem::path plan_directory(const std::filesystem::path& cache, int device);
    void select_matrix(MatrixPlan& plan, cublasLtHandle_t blas, cudaStream_t stream, const void* a, const void* b, const void* previous, float alpha, float beta, std::size_t output_bytes, void* workspace, std::size_t workspace_bytes, const std::filesystem::path& path);
} // namespace genesia::compute
