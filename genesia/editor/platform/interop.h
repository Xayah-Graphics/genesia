#ifndef GENESIA_EDITOR_INTEROP_H
#define GENESIA_EDITOR_INTEROP_H
#include <cstdint>
#include <genesia/cuda_stream.h>
namespace genesia::editor {
    void pack_rgba(::cuda::stream_ref stream, std::uint8_t* output, const std::uint8_t* input, int pixels);
}
#endif
