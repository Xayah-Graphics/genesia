module;
#include <genesia/cuda.h>
export module genesia.editor.platform.interop;
import genesia.editor.runtime.device;
import genesia.editor.runtime.resources;
import genesia.sdxl;
import std;
import vulkan;

export namespace genesia::editor {
    struct Interop final {
        struct Slot final {
            runtime::Buffer buffer;
            vk::raii::Semaphore timeline{nullptr};
            cudaExternalMemory_t memory{};
            cudaExternalSemaphore_t semaphore{};
            std::uint8_t* pixels{};
            std::uint64_t value{};
        };
        runtime::Device& device;
        std::array<Slot, 2> slots;
        explicit Interop(runtime::Device& device);
        ~Interop();
        std::uint64_t publish(const sdxl::Output& output, std::size_t slot);
    };
}
