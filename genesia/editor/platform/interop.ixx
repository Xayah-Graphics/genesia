module;
#include <genesia/cuda.h>
export module genesia.editor.platform.interop;
import genesia.editor.runtime.device;
import genesia.editor.runtime.resources;
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
        void prepare(int width, int height, ::cuda::stream_ref stream);
        std::uint64_t publish(const std::uint8_t* pixels, int width, int height, ::cuda::stream_ref stream, std::size_t slot);
    };
} // namespace genesia::editor
