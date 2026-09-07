module genesia.editor.runtime.resources;
import std;
import vulkan;

namespace genesia::editor::runtime {
    Buffer::Buffer(Device& device, const vk::DeviceSize bytes, const bool host, const vk::BufferUsageFlags extra, const bool external) : size{bytes} {
        const vk::ExternalMemoryBufferCreateInfo exported{vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32};
        buffer = vk::raii::Buffer{device.logical, vk::BufferCreateInfo{{}, bytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | extra, vk::SharingMode::eExclusive, 0, nullptr, external ? &exported : nullptr}};
        const auto requirement = buffer.getMemoryRequirements();
        allocation_size = requirement.size;
        const auto flags = host ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent : vk::MemoryPropertyFlags{vk::MemoryPropertyFlagBits::eDeviceLocal};
        std::uint32_t type{};
        for (std::uint32_t i = 0; i < device.memory.memoryTypeCount; ++i)
            if ((requirement.memoryTypeBits & (1u << i)) && (device.memory.memoryTypes[i].propertyFlags & flags) == flags) { type = i; break; }
        const vk::MemoryDedicatedAllocateInfo dedicated{{}, *buffer};
        const vk::ExportMemoryAllocateInfo export_memory{vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32, &dedicated};
        const vk::MemoryAllocateFlagsInfo allocation_flags{vk::MemoryAllocateFlagBits::eDeviceAddress, 0, external ? &export_memory : nullptr};
        memory = vk::raii::DeviceMemory{device.logical, vk::MemoryAllocateInfo{requirement.size, type, &allocation_flags}};
        buffer.bindMemory(*memory, 0);
        address = device.logical.getBufferAddress(vk::BufferDeviceAddressInfo{*buffer});
        if (host) mapped = memory.mapMemory(0, vk::WholeSize);
    }

    Buffer::~Buffer() {
        if (mapped) memory.unmapMemory();
    }

    Buffer::Buffer(Buffer&& other) noexcept : memory{std::move(other.memory)}, buffer{std::move(other.buffer)}, address{std::exchange(other.address, 0)}, size{std::exchange(other.size, 0)}, allocation_size{other.allocation_size}, mapped{std::exchange(other.mapped, nullptr)} {}

    Buffer& Buffer::operator=(Buffer&& other) noexcept {
        if (this == &other) return *this;
        if (mapped) memory.unmapMemory();
        buffer = nullptr;
        memory = std::move(other.memory);
        buffer = std::move(other.buffer);
        address = std::exchange(other.address, 0);
        size = std::exchange(other.size, 0);
        allocation_size = other.allocation_size;
        mapped = std::exchange(other.mapped, nullptr);
        return *this;
    }

    Image::Image(Device& device, const vk::Extent2D dimensions, const vk::Format type) : extent{dimensions}, format{type} {
        image = vk::raii::Image{device.logical, vk::ImageCreateInfo{{}, vk::ImageType::e2D, type, {extent.width, extent.height, 1}, 1, 1, vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive}};
        const auto requirement = image.getMemoryRequirements();
        std::uint32_t memory_type{};
        for (std::uint32_t i = 0; i < device.memory.memoryTypeCount; ++i)
            if ((requirement.memoryTypeBits & (1u << i)) && (device.memory.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) { memory_type = i; break; }
        memory = vk::raii::DeviceMemory{device.logical, vk::MemoryAllocateInfo{requirement.size, memory_type}};
        image.bindMemory(*memory, 0);
    }

    Resources::Resources(Device& gpu) : device{gpu} {
        const auto& props = device.heap_properties;
        const auto alignment = std::max(props.imageDescriptorAlignment, props.bufferDescriptorAlignment);
        resource_stride = (std::max(props.imageDescriptorSize, props.bufferDescriptorSize) + alignment - 1) & ~(alignment - 1);
        const auto resource_bytes = (resource_stride * 16384 + props.minResourceHeapReservedRange + props.resourceHeapAlignment - 1) & ~(props.resourceHeapAlignment - 1);
        const auto sampler_bytes = (props.samplerDescriptorSize * 64 + props.minSamplerHeapReservedRange + props.samplerHeapAlignment - 1) & ~(props.samplerHeapAlignment - 1);
        resource_heap = Buffer{gpu, resource_bytes, true, vk::BufferUsageFlagBits::eDescriptorHeapEXT};
        sampler_heap = Buffer{gpu, sampler_bytes, true, vk::BufferUsageFlagBits::eDescriptorHeapEXT};
    }

    void Resources::bind(const vk::raii::CommandBuffer& commands) const {
        const auto& props = device.heap_properties;
        commands.bindResourceHeapEXT(vk::BindHeapInfoEXT{{resource_heap.address, resource_heap.size}, resource_heap.size - props.minResourceHeapReservedRange, props.minResourceHeapReservedRange});
        commands.bindSamplerHeapEXT(vk::BindHeapInfoEXT{{sampler_heap.address, sampler_heap.size}, sampler_heap.size - props.minSamplerHeapReservedRange, props.minSamplerHeapReservedRange});
    }

    void Resources::describe(const std::uint32_t slot, const Buffer& buffer) {
        const vk::DeviceAddressRangeEXT address{buffer.address, buffer.size};
        device.logical.writeResourceDescriptorsEXT(vk::ResourceDescriptorInfoEXT{vk::DescriptorType::eStorageBuffer, vk::ResourceDescriptorDataEXT{&address}},
            vk::HostAddressRangeEXT{static_cast<std::byte*>(resource_heap.mapped) + slot * resource_stride, resource_stride});
    }

    void Resources::describe(const std::uint32_t slot, const Image& image) {
        const vk::ImageViewCreateInfo view{{}, *image.image, vk::ImageViewType::e2D, image.format, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        const vk::ImageDescriptorInfoEXT info{&view, vk::ImageLayout::eShaderReadOnlyOptimal};
        device.logical.writeResourceDescriptorsEXT(vk::ResourceDescriptorInfoEXT{vk::DescriptorType::eSampledImage, vk::ResourceDescriptorDataEXT{&info}},
            vk::HostAddressRangeEXT{static_cast<std::byte*>(resource_heap.mapped) + slot * resource_stride, resource_stride});
    }

}
