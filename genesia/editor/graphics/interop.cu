#include "interop.h"
#include <cuda/launch>
namespace genesia::editor {
    __global__ void pack_rgba_kernel(std::uint8_t* output, const std::uint8_t* input, const int pixels) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= pixels) return;
        reinterpret_cast<uchar4*>(output)[i] = {input[i * 3], input[i * 3 + 1], input[i * 3 + 2], 255};
    }
    cudaError_t prepare_rgba() {
        cudaFuncAttributes attributes{};
        return cudaFuncGetAttributes(&attributes, pack_rgba_kernel);
    }
    void pack_rgba(const ::cuda::stream_ref stream, std::uint8_t* output, const std::uint8_t* input, const int pixels) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((pixels + 255) / 256), ::cuda::block_dims(256))), pack_rgba_kernel, output, input, pixels);
    }
} // namespace genesia::editor
