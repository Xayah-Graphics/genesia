#ifndef GENESIA_SDXL_CONTROL_H
#define GENESIA_SDXL_CONTROL_H
#include <cstdint>

namespace genesia::sdxl {
    enum class Stage : std::uint32_t { idle, loading, preparing, sampling, decoding, transferring, complete, cancelled };
    struct Control final {
        alignas(64) std::uint32_t cancel{};
        alignas(64) std::uint32_t completed{};
        std::uint32_t stage{};
    };
}
#endif
