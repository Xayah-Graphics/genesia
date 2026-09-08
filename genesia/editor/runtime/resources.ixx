export module genesia.editor.runtime.resources;
import genesia.editor.runtime.device;
import std;
import vulkan;

export namespace genesia::editor::runtime {
    struct Buffer final {
        vk::raii::DeviceMemory memory{nullptr};
        vk::raii::Buffer buffer{nullptr};
        vk::DeviceAddress address{};
        vk::DeviceSize size{};
        vk::DeviceSize allocation_size{};
        void* mapped{};

        Buffer() = default;
        Buffer(Device& device, vk::DeviceSize size, bool host = false, vk::BufferUsageFlags extra = {}, bool external = false);
        ~Buffer();
        Buffer(Buffer&&) noexcept;
        Buffer& operator=(Buffer&&) noexcept;
        Buffer(const Buffer&)            = delete;
        Buffer& operator=(const Buffer&) = delete;
    };

    struct Image final {
        vk::raii::DeviceMemory memory{nullptr};
        vk::raii::Image image{nullptr};
        vk::Extent2D extent{};
        vk::Format format{};
        Image() = default;
        Image(Device& device, vk::Extent2D extent, vk::Format format);
    };

    struct Resources final {
        explicit Resources(Device& device);
        void bind(const vk::raii::CommandBuffer& commands) const;
        void describe(std::uint32_t slot, const Buffer& buffer);
        void describe(std::uint32_t slot, const Image& image);

        Device& device;
        Buffer resource_heap;
        Buffer sampler_heap;
        std::uint32_t resource_index{};
        vk::DeviceSize resource_stride{};
    };

} // namespace genesia::editor::runtime
