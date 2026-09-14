#ifndef GENESIA_EDITOR_INTEROP_H
#define GENESIA_EDITOR_INTEROP_H
#include <cstdint>
#include <cuda_runtime_api.h>
#include <genesia/cuda_stream.h>
namespace genesia::editor {
    cudaError_t prepare_rgba();
    void pack_rgba(::cuda::stream_ref stream, std::uint8_t* output, const std::uint8_t* input, int pixels);
} // namespace genesia::editor
#endif
