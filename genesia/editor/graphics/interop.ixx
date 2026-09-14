module;
#include <genesia/cuda.h>
export module genesia.editor.graphics.interop;
import genesia.editor.graphics.device;
import genesia.editor.graphics.resources;
import std;
import vulkan;

export namespace genesia::editor {
    struct Interop final {
        struct Slot final {
            graphics::Buffer buffer;
            vk::raii::Semaphore timeline{nullptr};
            std::unique_ptr<std::remove_pointer_t<cudaExternalMemory_t>, decltype(&cudaDestroyExternalMemory)> memory{nullptr, cudaDestroyExternalMemory};
            std::unique_ptr<std::remove_pointer_t<cudaExternalSemaphore_t>, decltype(&cudaDestroyExternalSemaphore)> semaphore{nullptr, cudaDestroyExternalSemaphore};
            std::unique_ptr<std::uint8_t, decltype(&cudaFree)> pixels{nullptr, cudaFree};
            std::uint64_t value{};
        };
        graphics::Device& device;
        std::array<Slot, 2> slots;
        explicit Interop(graphics::Device& device);
        ~Interop();
        void prepare(int width, int height, ::cuda::stream_ref stream);
        std::uint64_t publish(const std::uint8_t* pixels, int width, int height, ::cuda::stream_ref stream, std::size_t slot);
    };
} // namespace genesia::editor
