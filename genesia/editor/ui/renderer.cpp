module;
#include <Windows.h>

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
module genesia.editor.ui.renderer;
import genesia.editor.platform.window;
import genesia.editor.runtime.device;
import genesia.editor.runtime.resources;
import genesia.generation.output;
import std;
import vulkan;

namespace genesia::editor {
    Renderer::Renderer(WindowPlatform& platform) : window{platform}, instance{std::array{vk::KHRSurfaceExtensionName, vk::KHRWin32SurfaceExtensionName}}, surface{instance.instance, vk::Win32SurfaceCreateInfoKHR{{}, GetModuleHandleW(nullptr), window.native_window}}, device{instance}, resources{device} {
        pool     = vk::raii::CommandPool{device.logical, vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eResetCommandBuffer, device.family}};
        commands = vk::raii::CommandBuffers{device.logical, vk::CommandBufferAllocateInfo{*pool, vk::CommandBufferLevel::ePrimary, 2}};
        for (auto& frame : frames) {
            frame.available = vk::raii::Semaphore{device.logical, vk::SemaphoreCreateInfo{}};
            frame.finished  = vk::raii::Fence{device.logical, vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled}};
        }
        resources.resource_index = 4;
        std::array<std::vector<std::uint32_t>, 2> code;
        const std::array names{"imgui_vertex", "imgui_fragment"};
        std::array<vk::ShaderCreateInfoEXT, 2> info;
        for (int i = 0; i < 2; ++i) {
            std::ifstream file{std::filesystem::path{GENESIA_SHADER_DIRECTORY} / (std::string{names[i]} + ".spv"), std::ios::binary | std::ios::ate};
            file.exceptions(std::ios::badbit | std::ios::failbit);
            code[i].resize(static_cast<std::size_t>(file.tellg()) / 4);
            file.seekg(0);
            file.read(reinterpret_cast<char*>(code[i].data()), code[i].size() * 4);
            info[i] = {vk::ShaderCreateFlagBitsEXT::eLinkStage | vk::ShaderCreateFlagBitsEXT::eDescriptorHeap, i == 0 ? vk::ShaderStageFlagBits::eVertex : vk::ShaderStageFlagBits::eFragment, i == 0 ? vk::ShaderStageFlagBits::eFragment : vk::ShaderStageFlags{}, vk::ShaderCodeTypeEXT::eSpirv, code[i].size() * 4, code[i].data(), names[i]};
        }
        shaders = vk::raii::ShaderEXTs{device.logical, info};
        device.logical.writeSamplerDescriptorsEXT(vk::SamplerCreateInfo{{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge}, vk::HostAddressRangeEXT{resources.sampler_heap.mapped, device.heap_properties.samplerDescriptorSize});
        ImGui::CreateContext();
        auto& io               = ImGui::GetIO();
        io.IniFilename         = nullptr;
        io.BackendRendererName = "genesia_shader_object";
        io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/SegUIVar.ttf");
        if (!ImGui_ImplGlfw_InitForVulkan(window.window, true)) throw std::runtime_error{"Cannot initialize ImGui input"};
        recreate();
    }

    Renderer::~Renderer() {
        device.logical.waitIdle();
        for (auto* texture : ImGui::GetPlatformIO().Textures) {
            texture->BackendUserData = nullptr;
            texture->SetTexID(ImTextureID_Invalid);
            texture->SetStatus(ImTextureStatus_Destroyed);
        }
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    bool Renderer::begin() {
        int width, height;
        glfwGetFramebufferSize(window.window, &width, &height);
        visible = width != 0 && height != 0;
        if (visible && (extent.width != width || extent.height != height)) recreate();
        frame_started = std::chrono::steady_clock::now();
        auto& frame   = frames[frame_index];
        static_cast<void>(device.logical.waitForFences(*frame.finished, true, std::numeric_limits<std::uint64_t>::max()));
        frame.uploads.clear();
        frame.retired.clear();
        free_descriptors.insert(free_descriptors.end(), frame.recycled.begin(), frame.recycled.end());
        frame.recycled.clear();
        try {
            if (visible) {
                const auto acquired = swapchain.acquireNextImage(std::numeric_limits<std::uint64_t>::max(), *frame.available);
                image_index         = acquired.value;
            }
        } catch (const vk::OutOfDateKHRError&) {
            recreate();
            return false;
        }
        device.logical.resetFences(*frame.finished);
        commands[frame_index].reset();
        commands[frame_index].begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        waits.clear();
        signals.clear();
        float scale, vertical;
        glfwGetWindowContentScale(window.window, &scale, &vertical);
        if (scale != dpi) ImGui::GetStyle().ScaleAllSizes(scale / dpi);
        dpi                            = scale;
        ImGui::GetStyle().FontSizeBase = 15.5F;
        ImGui::GetStyle().FontScaleDpi = dpi;
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        return true;
    }

    void Renderer::present() {
        ImGui::Render();
        update_fonts();
        if (visible) draw();
        const auto& command = commands[frame_index];
        command.end();
        if (visible) {
            waits.emplace_back(*frames[frame_index].available, 0, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
            signals.emplace_back(*rendered[image_index], 0, vk::PipelineStageFlagBits2::eAllCommands);
        }
        const vk::CommandBufferSubmitInfo submitted{*command};
        device.graphics.submit2(vk::SubmitInfo2{{}, static_cast<std::uint32_t>(waits.size()), waits.data(), 1, &submitted, static_cast<std::uint32_t>(signals.size()), signals.data()}, *frames[frame_index].finished);
        const auto semaphore = *rendered[image_index];
        const auto chain     = *swapchain;
        try {
            if (visible) static_cast<void>(device.graphics.presentKHR(vk::PresentInfoKHR{1, &semaphore, 1, &chain, &image_index}));
        } catch (const vk::OutOfDateKHRError&) {
            extent = vk::Extent2D{};
        }
        frame_index        = (frame_index + 1) % frames.size();
        last_frame_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - frame_started).count();
    }

    std::uint64_t Renderer::texture(const vk::Extent2D extent, const vk::Format format) {
        const auto slot = free_descriptors.empty() ? resources.resource_index++ : free_descriptors.back();
        if (!free_descriptors.empty()) free_descriptors.pop_back();
        const std::uint64_t id = slot + 1;
        auto& value            = textures.emplace(id, Texture{runtime::Image{device, extent, format}}).first->second;
        resources.describe(slot, value.image);
        return id;
    }

    std::uint64_t Renderer::upload(const Image& image) {
        std::vector<std::uint8_t> rgba(std::size_t(image.width) * image.height * 4);
        for (std::size_t i = 0; i < rgba.size() / 4; ++i) {
            std::memcpy(rgba.data() + i * 4, image.pixels.data() + i * 3, 3);
            rgba[i * 4 + 3] = 255;
        }
        const auto id = texture({std::uint32_t(image.width), std::uint32_t(image.height)});
        upload(id, rgba.data(), image.width, image.height, true);
        return id;
    }

    void Renderer::upload(const std::uint64_t id, const void* rgba, const int width, const int height, const bool initial) {
        auto& image   = textures.at(id).image;
        auto& staging = frames[frame_index].uploads.emplace_back(device, std::size_t(width) * height * 4, true);
        std::memcpy(staging.mapped, rgba, staging.size);
        const auto& command = commands[frame_index];
        const vk::ImageMemoryBarrier2 transfer{initial ? vk::PipelineStageFlagBits2::eNone : vk::PipelineStageFlagBits2::eFragmentShader, initial ? vk::AccessFlags2{} : vk::AccessFlagBits2::eShaderSampledRead, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite, initial ? vk::ImageLayout::eUndefined : vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, *image.image, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        command.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, {}, {}, 1, &transfer});
        command.copyBufferToImage(*staging.buffer, *image.image, vk::ImageLayout::eTransferDstOptimal, vk::BufferImageCopy{0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {std::uint32_t(width), std::uint32_t(height), 1}});
        const vk::ImageMemoryBarrier2 sampled{vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, *image.image, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        command.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, {}, {}, 1, &sampled});
        textures.at(id).initialized = true;
    }

    void Renderer::copy(const std::uint64_t id, const runtime::Buffer& source, const vk::Semaphore semaphore, const std::uint64_t ready) {
        auto& target        = textures.at(id);
        const auto& command = commands[frame_index];
        waits.emplace_back(semaphore, ready, vk::PipelineStageFlagBits2::eCopy);
        signals.emplace_back(semaphore, ready + 1, vk::PipelineStageFlagBits2::eAllCommands);
        const vk::BufferMemoryBarrier2 acquire{vk::PipelineStageFlagBits2::eNone, {}, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead, vk::QueueFamilyExternal, device.family, *source.buffer, 0, source.size};
        const vk::ImageMemoryBarrier2 transfer{target.initialized ? vk::PipelineStageFlagBits2::eFragmentShader : vk::PipelineStageFlagBits2::eNone, target.initialized ? vk::AccessFlagBits2::eShaderSampledRead : vk::AccessFlags2{}, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite, target.initialized ? vk::ImageLayout::eShaderReadOnlyOptimal : vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, *target.image.image, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        command.pipelineBarrier2(vk::DependencyInfo{{}, 0, nullptr, 1, &acquire, 1, &transfer});
        command.copyBufferToImage(*source.buffer, *target.image.image, vk::ImageLayout::eTransferDstOptimal, vk::BufferImageCopy{0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {target.image.extent.width, target.image.extent.height, 1}});
        const vk::BufferMemoryBarrier2 release{vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eNone, {}, device.family, vk::QueueFamilyExternal, *source.buffer, 0, source.size};
        const vk::ImageMemoryBarrier2 sampled{vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, *target.image.image, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        command.pipelineBarrier2(vk::DependencyInfo{{}, 0, nullptr, 1, &release, 1, &sampled});
        target.initialized = true;
    }

    void Renderer::discard(const vk::Semaphore semaphore, const std::uint64_t ready) {
        // A dropped frame still returns its external buffer to the CUDA producer.
        waits.emplace_back(semaphore, ready, vk::PipelineStageFlagBits2::eAllCommands);
        signals.emplace_back(semaphore, ready + 1, vk::PipelineStageFlagBits2::eAllCommands);
    }

    void Renderer::retire(const std::uint64_t id) {
        auto node = textures.extract(id);
        frames[frame_index].retired.push_back(std::move(node.mapped()));
        frames[frame_index].recycled.push_back(static_cast<std::uint32_t>(id - 1));
    }

    void Renderer::recreate() {
        device.logical.waitIdle();
        int width, height;
        glfwGetFramebufferSize(window.window, &width, &height);
        extent = vk::Extent2D{std::uint32_t(width), std::uint32_t(height)};
        const vk::SwapchainCreateInfoKHR info{{}, *surface, 3, vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear, extent, 1, vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive, 0, nullptr, vk::SurfaceTransformFlagBitsKHR::eIdentity, vk::CompositeAlphaFlagBitsKHR::eOpaque, vk::PresentModeKHR::eMailbox, true, *swapchain};
        vk::raii::SwapchainKHR replacement{device.logical, info};
        views.clear();
        rendered.clear();
        swapchain = std::move(replacement);
        images    = swapchain.getImages();
        for (const auto image : images) {
            views.emplace_back(device.logical, vk::ImageViewCreateInfo{{}, image, vk::ImageViewType::e2D, vk::Format::eB8G8R8A8Srgb, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
            rendered.emplace_back(device.logical, vk::SemaphoreCreateInfo{});
        }
    }

    void Renderer::update_fonts() {
        for (auto* font : ImGui::GetPlatformIO().Textures) {
            if (font->Status == ImTextureStatus_WantDestroy && font->UnusedFrames >= 2) {
                retire(font->GetTexID());
                font->SetTexID(ImTextureID_Invalid);
                font->SetStatus(ImTextureStatus_Destroyed);
            } else if (font->Status == ImTextureStatus_WantCreate || font->Status == ImTextureStatus_WantUpdates) {
                const bool initial = font->Status == ImTextureStatus_WantCreate;
                if (initial) font->SetTexID(texture({std::uint32_t(font->Width), std::uint32_t(font->Height)}, vk::Format::eR8G8B8A8Unorm));
                upload(font->GetTexID(), font->Pixels, font->Width, font->Height, initial);
                font->SetStatus(ImTextureStatus_OK);
            }
        }
    }

    void Renderer::draw() {
        const auto& command = commands[frame_index];
        auto& frame         = frames[frame_index];
        auto& data          = *ImGui::GetDrawData();
        if (data.TotalVtxCount) {
            const auto vertex_bytes = std::size_t(data.TotalVtxCount) * sizeof(ImDrawVert);
            const auto index_bytes  = std::size_t(data.TotalIdxCount) * 4;
            if (frame.vertices.size < vertex_bytes) frame.vertices = runtime::Buffer{device, std::bit_ceil(vertex_bytes), true};
            if (frame.indices.size < index_bytes) frame.indices = runtime::Buffer{device, std::bit_ceil(index_bytes), true};
            resources.describe(std::uint32_t(frame_index * 2), frame.vertices);
            resources.describe(std::uint32_t(frame_index * 2 + 1), frame.indices);
            auto* vertices = static_cast<std::byte*>(frame.vertices.mapped);
            auto* indices  = static_cast<std::uint32_t*>(frame.indices.mapped);
            for (const auto* list : data.CmdLists) {
                std::memcpy(vertices, list->VtxBuffer.Data, list->VtxBuffer.Size * sizeof(ImDrawVert));
                vertices += list->VtxBuffer.Size * sizeof(ImDrawVert);
                for (const auto index : list->IdxBuffer) *indices++ = index;
            }
        }
        const vk::ImageMemoryBarrier2 attachment{vk::PipelineStageFlagBits2::eNone, {}, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, images[image_index], {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        command.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, {}, {}, 1, &attachment});
        const vk::RenderingAttachmentInfo color{*views[image_index], vk::ImageLayout::eColorAttachmentOptimal, {}, {}, {}, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::ClearValue{vk::ClearColorValue{std::array{0.0056F, 0.0060F, 0.0080F, 1.0F}}}};
        command.beginRendering(vk::RenderingInfo{{}, {{0, 0}, extent}, 1, 0, 1, &color});
        command.setViewportWithCount(vk::Viewport{0, 0, float(extent.width), float(extent.height), 0, 1});
        command.setCullMode(vk::CullModeFlagBits::eNone);
        command.setFrontFace(vk::FrontFace::eCounterClockwise);
        command.setDepthTestEnable(false);
        command.setDepthWriteEnable(false);
        command.setDepthCompareOp(vk::CompareOp::eAlways);
        command.setRasterizerDiscardEnable(false);
        command.setPolygonModeEXT(vk::PolygonMode::eFill);
        command.setRasterizationSamplesEXT(vk::SampleCountFlagBits::e1);
        command.setAlphaToCoverageEnableEXT(false);
        command.setDepthBiasEnable(false);
        command.setStencilTestEnable(false);
        command.setSampleMaskEXT(vk::SampleCountFlagBits::e1, vk::SampleMask{1});
        command.setVertexInputEXT({}, {});
        command.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);
        command.setPrimitiveRestartEnable(false);
        command.setColorBlendEnableEXT(0, vk::Bool32{true});
        command.setColorBlendEquationEXT(0, vk::ColorBlendEquationEXT{vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd});
        command.setColorWriteMaskEXT(0, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        command.bindShadersEXT(std::array{vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment}, std::array{*shaders[0], *shaders[1]});
        resources.bind(command);
        std::uint32_t vertex_offset{}, index_offset{};
        for (const auto* list : data.CmdLists) {
            for (const auto& draw : list->CmdBuffer) {
                if (draw.UserCallback) {
                    if (draw.UserCallback != ImDrawCallback_ResetRenderState) draw.UserCallback(list, &draw);
                    continue;
                }
                const float x      = std::max(0.0F, (draw.ClipRect.x - data.DisplayPos.x) * data.FramebufferScale.x);
                const float y      = std::max(0.0F, (draw.ClipRect.y - data.DisplayPos.y) * data.FramebufferScale.y);
                const float right  = std::min(float(extent.width), (draw.ClipRect.z - data.DisplayPos.x) * data.FramebufferScale.x);
                const float bottom = std::min(float(extent.height), (draw.ClipRect.w - data.DisplayPos.y) * data.FramebufferScale.y);
                if (right <= x || bottom <= y) continue;
                command.setScissorWithCount(vk::Rect2D{{int(x), int(y)}, {std::uint32_t(right - x), std::uint32_t(bottom - y)}});
                const struct Push {
                    std::array<std::uint32_t, 2> vertices, indices, texture, sampler;
                    std::uint32_t index_offset, vertex_offset;
                    std::array<float, 2> scale, translation;
                } push{{std::uint32_t(frame_index * 2), 0}, {std::uint32_t(frame_index * 2 + 1), 0}, {std::uint32_t(draw.GetTexID() - 1), 0}, {0, 0}, index_offset + draw.IdxOffset, vertex_offset + draw.VtxOffset, {2 / data.DisplaySize.x, 2 / data.DisplaySize.y}, {-1 - data.DisplayPos.x * 2 / data.DisplaySize.x, -1 - data.DisplayPos.y * 2 / data.DisplaySize.y}};
                command.pushDataEXT(vk::PushDataInfoEXT{0, vk::HostAddressRangeConstEXT{&push, sizeof(push)}});
                command.draw(draw.ElemCount, 1, 0, 0);
            }
            vertex_offset += list->VtxBuffer.Size;
            index_offset += list->IdxBuffer.Size;
        }
        command.endRendering();
        const vk::ImageMemoryBarrier2 presented{vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eNone, {}, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, images[image_index], {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        command.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, {}, {}, 1, &presented});
    }
} // namespace genesia::editor
