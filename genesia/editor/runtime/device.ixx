export module genesia.editor.runtime.device;
import std;
import vulkan;

export namespace genesia::editor::runtime {
    struct Instance final {
        explicit Instance(std::span<const char* const> extensions);
        vk::raii::Context context;
        vk::raii::Instance instance{nullptr};
    };

    struct Device final {
        explicit Device(Instance& instance);
        ~Device();
        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;

        vk::raii::PhysicalDevice physical{nullptr};
        vk::raii::Device logical{nullptr};
        vk::raii::Queue graphics{nullptr};
        std::uint32_t family{};
        vk::PhysicalDeviceMemoryProperties memory;
        vk::PhysicalDeviceDescriptorHeapPropertiesEXT heap_properties;
    };
}
