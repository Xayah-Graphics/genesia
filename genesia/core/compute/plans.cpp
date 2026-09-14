module;
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend_version.h>
#include <nlohmann/json.hpp>
module genesia.compute.plans;
import genesia.io.files;
import std;
namespace genesia::compute {
    Handles::Handles(const cudaStream_t stream) {
        check(cublasLtCreate(std::out_ptr(blas)));
        check(cudnnCreate(std::out_ptr(dnn)));
        check(cudnnSetStream(dnn.get(), stream));
    }
    std::filesystem::path plan_directory(const std::filesystem::path& cache, const int device) {
        cudaDeviceProp properties;
        check(cudaGetDeviceProperties(&properties, device));
        int driver;
        check(cudaDriverGetVersion(&driver));
        std::string identity;
        for (const auto byte : properties.uuid.bytes) identity += std::format("{:02x}", static_cast<unsigned char>(byte));
        const auto path = cache / "compute" / std::format("v1-{}-{}-{}-{}-{}-{}-{}-{}", identity, GENESIA_PLATFORM, CUDART_VERSION, driver, cudnnGetVersion(), CUDNN_FRONTEND_VERSION, cublasLtGetVersion(), GENESIA_CXX_COMPILER);
        std::filesystem::create_directories(path);
        return path;
    }
    void select_matrix(MatrixPlan& plan, const cublasLtHandle_t blas, const cudaStream_t stream, const void* a, const void* b, const void* previous, const float alpha, const float beta, const std::size_t output_bytes, void* workspace, const std::size_t workspace_bytes, const std::filesystem::path& path) {
        if (std::filesystem::exists(path)) {
            const auto cached = files::read_json(path);
            const auto bytes  = cached.at("algorithm").get<std::array<std::uint8_t, sizeof(cublasLtMatmulAlgo_t)>>();
            std::memcpy(&plan.algorithm, bytes.data(), bytes.size());
            plan.workspace_bytes = cached.at("workspace");
            return;
        }
        DeviceBuffer temporary{output_bytes};
        std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)> start{nullptr, cudaEventDestroy}, stop{nullptr, cudaEventDestroy};
        check(cudaEventCreate(std::out_ptr(start)));
        check(cudaEventCreate(std::out_ptr(stop)));
        std::unique_ptr<std::remove_pointer_t<cublasLtMatmulPreference_t>, decltype(&cublasLtMatmulPreferenceDestroy)> preference{nullptr, cublasLtMatmulPreferenceDestroy};
        check(cublasLtMatmulPreferenceCreate(std::out_ptr(preference)));
        const std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        check(cublasLtMatmulPreferenceSetAttribute(preference.get(), CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
        check(cublasLtMatmulPreferenceSetAttribute(preference.get(), CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction, sizeof(reduction)));
        std::array<cublasLtMatmulHeuristicResult_t, 32> candidates;
        int count{};
        check(cublasLtMatmulAlgoGetHeuristic(blas, plan.operation.get(), plan.a.get(), plan.b.get(), plan.output.get(), plan.output.get(), preference.get(), int(candidates.size()), candidates.data(), &count));
        float best = std::numeric_limits<float>::infinity();
        for (int i = 0; i < count; ++i) {
            if (candidates[i].state != CUBLAS_STATUS_SUCCESS) continue;
            const auto& algorithm = candidates[i].algo;
            cublasLtMatmulHeuristicResult_t checked{};
            if (cublasLtMatmulAlgoCheck(blas, plan.operation.get(), plan.a.get(), plan.b.get(), plan.output.get(), plan.output.get(), &algorithm, &checked) != CUBLAS_STATUS_SUCCESS || checked.state != CUBLAS_STATUS_SUCCESS) continue;
            check(cublasLtMatmul(blas, plan.operation.get(), &alpha, a, plan.a.get(), b, plan.b.get(), &beta, previous, plan.output.get(), temporary.data, plan.output.get(), &algorithm, workspace, workspace_bytes, stream));
            check(cudaEventRecord(start.get(), stream));
            for (int repeat = 0; repeat < 3; ++repeat) check(cublasLtMatmul(blas, plan.operation.get(), &alpha, a, plan.a.get(), b, plan.b.get(), &beta, previous, plan.output.get(), temporary.data, plan.output.get(), &algorithm, workspace, workspace_bytes, stream));
            check(cudaEventRecord(stop.get(), stream));
            check(cudaEventSynchronize(stop.get()));
            float elapsed;
            check(cudaEventElapsedTime(&elapsed, start.get(), stop.get()));
            if (elapsed < best) {
                best                 = elapsed;
                plan.algorithm       = algorithm;
                plan.workspace_bytes = checked.workspaceSize;
            }
        }
        if (!std::isfinite(best)) throw std::runtime_error{"No executable matrix algorithm"};
        std::array<std::uint8_t, sizeof(cublasLtMatmulAlgo_t)> bytes;
        std::memcpy(bytes.data(), &plan.algorithm, bytes.size());
        files::write_json(path, {{"algorithm", bytes}, {"workspace", plan.workspace_bytes}});
    }
} // namespace genesia::compute
