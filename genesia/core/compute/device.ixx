module;
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cudnn.h>
export module genesia.compute.device;
import std;
export namespace genesia::compute {
    void check(cudaError_t status);
    void check(cublasStatus_t status);
    void check(cudnnStatus_t status);
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
} // namespace genesia::compute
