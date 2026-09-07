#ifndef GENESIA_SDXL_CONTROL_H
#define GENESIA_SDXL_CONTROL_H
#include <cstdint>

namespace genesia::sdxl {
    // CPU: free -> requested, ready -> reading -> free. GPU: requested -> ready.
    // Cross-device handoffs use naturally aligned 32-bit release/acquire operations.
    enum class SnapshotState : std::uint32_t { free, requested, ready, reading };
    struct SnapshotSlot final {
        alignas(64) std::uint32_t state{};
        std::uint32_t step{};
    };

    enum class Stage : std::uint32_t { idle, loading, preparing, sampling, decoding, transferring, complete, cancelled };
    struct Control final {
        alignas(64) std::uint32_t cancel{};
        alignas(64) std::uint32_t completed{};
        std::uint32_t stage{};
    };
} // namespace genesia::sdxl
#endif
