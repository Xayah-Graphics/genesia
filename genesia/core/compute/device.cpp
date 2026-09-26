module;
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cudnn.h>
module genesia.compute.device;
import std;
namespace genesia::compute {
    void check(const cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error{cudaGetErrorString(status)};
    }
    void check(const cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt status {}", static_cast<int>(status))};
    }
    void check(const cudnnStatus_t status) {
        if (status != CUDNN_STATUS_SUCCESS) throw std::runtime_error{cudnnGetErrorString(status)};
    }
    DeviceBuffer::DeviceBuffer(std::size_t bytes) : size(bytes) {
        if (bytes) check(cudaMalloc(&data, bytes));
    }
    DeviceBuffer::~DeviceBuffer() {
        if (data) cudaFree(data);
    }
    DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : data(std::exchange(other.data, nullptr)), size(std::exchange(other.size, 0)) {}
    DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
        if (data) cudaFree(data);
        data = std::exchange(other.data, nullptr);
        size = std::exchange(other.size, 0);
        return *this;
    }
} // namespace genesia::compute
