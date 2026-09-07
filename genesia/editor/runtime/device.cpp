module genesia.editor.runtime.device;
import std;
import vulkan;

namespace genesia::editor::runtime {
    Instance::Instance(const std::span<const char* const> extensions) {
        const vk::ApplicationInfo application{"Genesia", 1, "Genesia", 1, vk::ApiVersion14};
        instance = vk::raii::Instance{context, vk::InstanceCreateInfo{{}, &application, 0, nullptr, static_cast<std::uint32_t>(extensions.size()), extensions.data()}};
    }

    Device::Device(Instance& instance) {
        for (const auto& candidate : instance.instance.enumeratePhysicalDevices()) {
            const auto info = candidate.getProperties();
            if (info.vendorID == 0x10de && info.deviceID == 0x2b85) physical = candidate;
        }
        memory = physical.getMemoryProperties();
        const auto extended = physical.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();
        heap_properties = extended.get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();
        const auto families = physical.getQueueFamilyProperties();
        for (std::uint32_t i = 0; i < families.size(); ++i)
            if ((families[i].queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) == (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) { family = i; break; }
        std::vector<const char*> extensions{
            vk::EXTDescriptorHeapExtensionName, vk::KHRShaderUntypedPointersExtensionName, vk::EXTShaderObjectExtensionName,
            vk::KHRExternalMemoryWin32ExtensionName, vk::KHRExternalSemaphoreWin32ExtensionName,
            vk::KHRSwapchainExtensionName, vk::KHRSwapchainMaintenance1ExtensionName};
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceDescriptorHeapFeaturesEXT, vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR,
            vk::PhysicalDeviceShaderObjectFeaturesEXT, vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR> features;
        features.get<vk::PhysicalDeviceFeatures2>().features.shaderInt64 = true;
        features.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters = true;
        auto& v12 = features.get<vk::PhysicalDeviceVulkan12Features>();
        v12.bufferDeviceAddress = v12.scalarBlockLayout = v12.timelineSemaphore = true;
        auto& v13 = features.get<vk::PhysicalDeviceVulkan13Features>();
        v13.synchronization2 = v13.dynamicRendering = true;
        features.get<vk::PhysicalDeviceDescriptorHeapFeaturesEXT>().descriptorHeap = true;
        features.get<vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR>().shaderUntypedPointers = true;
        features.get<vk::PhysicalDeviceShaderObjectFeaturesEXT>().shaderObject = true;
        features.get<vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR>().swapchainMaintenance1 = true;
        const std::array priorities{1.0F};
        const vk::DeviceQueueCreateInfo queues{{}, family, 1, priorities.data()};
        logical = vk::raii::Device{physical, vk::DeviceCreateInfo{{}, 1, &queues, 0, nullptr, static_cast<std::uint32_t>(extensions.size()), extensions.data(), nullptr, &features.get<vk::PhysicalDeviceFeatures2>()}};
        graphics = logical.getQueue(family, 0);
    }

    Device::~Device() {
        if (*logical) logical.waitIdle();
    }
}
