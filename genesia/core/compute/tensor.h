#ifndef GENESIA_COMPUTE_TENSOR_H
#define GENESIA_COMPUTE_TENSOR_H
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
namespace genesia::compute {
    enum class Scalar : std::uint8_t { f32, f16, bf16 };
    struct TensorView final {
        void* data{};
        int n{1}, h{1}, w{1}, c{1};
        Scalar scalar{Scalar::f16};
        __host__ __device__ std::size_t elements() const {
            return std::size_t(n) * h * w * c;
        }
        __host__ __device__ std::size_t bytes() const {
            return elements() * (scalar == Scalar::f32 ? 4 : 2);
        }
        __host__ __device__ TensorView reshape(int batch, int height, int width, int channels, Scalar type = Scalar::f16) const {
            return {data, batch, height, width, channels, type};
        }
    };
} // namespace genesia::compute
#endif
