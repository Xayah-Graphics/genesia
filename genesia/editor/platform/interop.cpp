module;
#include <Windows.h>
#include <genesia/cuda.h>
#include "interop.h"
module genesia.editor.platform.interop;
import genesia.neural.inference_runtime;
import genesia.sdxl;
import std;
import vulkan;

namespace genesia::editor {
    Interop::Interop(runtime::Device& device) : device{device} {
        const vk::SemaphoreTypeCreateInfo type{vk::SemaphoreType::eTimeline};
        const vk::ExportSemaphoreCreateInfo exported{vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32, &type};
        for (auto& slot : slots) {
            slot.timeline = vk::raii::Semaphore{device.logical, vk::SemaphoreCreateInfo{{}, &exported}};
            const HANDLE handle = device.logical.getSemaphoreWin32HandleKHR(vk::SemaphoreGetWin32HandleInfoKHR{*slot.timeline, vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32});
            cudaExternalSemaphoreHandleDesc description{};
            description.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
            description.handle.win32.handle = handle;
            const auto status = cudaImportExternalSemaphore(&slot.semaphore, &description);
            CloseHandle(handle);
            neural::check(status);
        }
    }

    Interop::~Interop() {
        device.logical.waitIdle();
        for (auto& slot : slots) {
            if (slot.pixels) cudaFree(slot.pixels);
            if (slot.memory) cudaDestroyExternalMemory(slot.memory);
            cudaDestroyExternalSemaphore(slot.semaphore);
        }
    }

    std::uint64_t Interop::publish(const sdxl::Output& output, const std::size_t index) {
        auto& slot = slots[index];
        if (slot.value) {
            cudaExternalSemaphoreWaitParams wait{};
            wait.params.fence.value = slot.value + 1;
            neural::check(cudaWaitExternalSemaphoresAsync(&slot.semaphore, &wait, 1, output.stream.get()));
        }
        const auto bytes = std::size_t(output.width) * output.height * 4;
        if (slot.buffer.size < bytes) {
            output.stream.sync();
            if (slot.pixels) neural::check(cudaFree(slot.pixels));
            if (slot.memory) neural::check(cudaDestroyExternalMemory(slot.memory));
            slot.buffer = runtime::Buffer{device, bytes, false, {}, true};
            const HANDLE handle = device.logical.getMemoryWin32HandleKHR(vk::MemoryGetWin32HandleInfoKHR{*slot.buffer.memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32});
            cudaExternalMemoryHandleDesc description{};
            description.type = cudaExternalMemoryHandleTypeOpaqueWin32;
            description.handle.win32.handle = handle;
            description.size = slot.buffer.allocation_size;
            description.flags = cudaExternalMemoryDedicated;
            const auto status = cudaImportExternalMemory(&slot.memory, &description);
            CloseHandle(handle);
            neural::check(status);
            const cudaExternalMemoryBufferDesc mapping{0, slot.buffer.allocation_size, 0};
            neural::check(cudaExternalMemoryGetMappedBuffer(reinterpret_cast<void**>(&slot.pixels), slot.memory, &mapping));
        }
        pack_rgba(output.stream, slot.pixels, output.device_pixels, output.width * output.height);
        slot.value = slot.value ? slot.value + 2 : 1;
        cudaExternalSemaphoreSignalParams signal{};
        signal.params.fence.value = slot.value;
        neural::check(cudaSignalExternalSemaphoresAsync(&slot.semaphore, &signal, 1, output.stream.get()));
        return slot.value;
    }
}
