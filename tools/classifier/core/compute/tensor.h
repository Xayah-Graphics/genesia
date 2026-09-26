#ifndef CLASSIFIER_COMPUTE_TENSOR_H
#define CLASSIFIER_COMPUTE_TENSOR_H
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
namespace classifier::compute {
    enum class Scalar : std::uint8_t { f32, bf16 };
    struct TensorView final {
        void* data{};
        int n{1}, h{1}, w{1}, c{1};
        Scalar scalar{Scalar::bf16};
        __host__ __device__ std::size_t elements() const {
            return std::size_t(n) * h * w * c;
        }
        __host__ __device__ std::size_t bytes() const {
            return elements() * (scalar == Scalar::f32 ? 4 : 2);
        }
    };
} // namespace classifier::compute
#endif
