export module tools.editor.graphics.resources;
import tools.editor.graphics.device;
import std;
import vulkan;

export namespace tools::editor::graphics {
    struct Buffer final {
        vk::raii::DeviceMemory memory{nullptr};
        vk::raii::Buffer buffer{nullptr};
        vk::DeviceAddress address{};
        vk::DeviceSize size{};
        void* mapped{};

        Buffer() = default;
        Buffer(Device& device, vk::DeviceSize size, vk::BufferUsageFlags extra = {});
        ~Buffer();
        Buffer(Buffer&&) noexcept;
        Buffer& operator=(Buffer&&) noexcept;
        Buffer(const Buffer&)            = delete;
        Buffer& operator=(const Buffer&) = delete;
    };

    struct Image final {
        vk::raii::DeviceMemory memory{nullptr};
        vk::raii::Image image{nullptr};
        Image() = default;
        Image(Device& device, vk::Extent2D extent);
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

} // namespace tools::editor::graphics
