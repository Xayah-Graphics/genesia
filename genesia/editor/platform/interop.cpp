module;
#include "interop.h"

#include <Windows.h>

#include <genesia/cuda.h>
module genesia.editor.platform.interop;
import genesia.neural.inference_runtime;
import std;
import vulkan;

namespace genesia::editor {
    Interop::Interop(runtime::Device& device) : device{device} {
        // First-use kernel loading must finish before sampling and preview overlap.
        neural::check(prepare_rgba());
        const vk::SemaphoreTypeCreateInfo type{vk::SemaphoreType::eTimeline};
        const vk::ExportSemaphoreCreateInfo exported{vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32, &type};
        for (auto& slot : slots) {
            slot.timeline = vk::raii::Semaphore{device.logical, vk::SemaphoreCreateInfo{{}, &exported}};
            const std::unique_ptr<void, decltype(&CloseHandle)> handle{device.logical.getSemaphoreWin32HandleKHR(vk::SemaphoreGetWin32HandleInfoKHR{*slot.timeline, vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32}), CloseHandle};
            cudaExternalSemaphoreHandleDesc description{};
            description.type                = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
            description.handle.win32.handle = handle.get();
            neural::check(cudaImportExternalSemaphore(std::out_ptr(slot.semaphore), &description));
        }
    }

    Interop::~Interop() {
        device.logical.waitIdle();
    }

    void Interop::prepare(const int width, const int height, const ::cuda::stream_ref stream) {
        const auto bytes = std::size_t(width) * height * 4;
        for (auto& slot : slots) {
            if (slot.buffer.size >= bytes) continue;
            if (slot.value) {
                cudaExternalSemaphoreWaitParams wait{};
                wait.params.fence.value = slot.value + 1;
                const auto semaphore    = slot.semaphore.get();
                neural::check(cudaWaitExternalSemaphoresAsync(&semaphore, &wait, 1, stream.get()));
            }
            stream.sync();
            slot.pixels.reset();
            slot.memory.reset();
            slot.buffer = runtime::Buffer{device, bytes, false, {}, true};
            const std::unique_ptr<void, decltype(&CloseHandle)> handle{device.logical.getMemoryWin32HandleKHR(vk::MemoryGetWin32HandleInfoKHR{*slot.buffer.memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32}), CloseHandle};
            cudaExternalMemoryHandleDesc description{};
            description.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
            description.handle.win32.handle = handle.get();
            description.size                = slot.buffer.allocation_size;
            description.flags               = cudaExternalMemoryDedicated;
            neural::check(cudaImportExternalMemory(std::out_ptr(slot.memory), &description));
            const cudaExternalMemoryBufferDesc mapping{0, slot.buffer.allocation_size, 0};
            neural::check(cudaExternalMemoryGetMappedBuffer(std::out_ptr<void*>(slot.pixels), slot.memory.get(), &mapping));
        }
    }

    std::uint64_t Interop::publish(const std::uint8_t* pixels, const int width, const int height, const ::cuda::stream_ref stream, const std::size_t index) {
        auto& slot           = slots[index];
        const auto semaphore = slot.semaphore.get();
        if (slot.value) {
            cudaExternalSemaphoreWaitParams wait{};
            wait.params.fence.value = slot.value + 1;
            neural::check(cudaWaitExternalSemaphoresAsync(&semaphore, &wait, 1, stream.get()));
        }
        pack_rgba(stream, slot.pixels.get(), pixels, width * height);
        slot.value = slot.value ? slot.value + 2 : 1;
        cudaExternalSemaphoreSignalParams signal{};
        signal.params.fence.value = slot.value;
        neural::check(cudaSignalExternalSemaphoresAsync(&semaphore, &signal, 1, stream.get()));
        return slot.value;
    }
} // namespace genesia::editor
