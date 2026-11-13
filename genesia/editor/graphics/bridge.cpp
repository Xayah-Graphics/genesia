module;
#include <GLFW/glfw3.h>
#include <genesia/cuda.h>
module genesia.editor.graphics.bridge;
import genesia.lan;
import std;
import vulkan;
namespace genesia::editor {
    WorkspaceRuntime::WorkspaceRuntime(graphics::Device& device)
        : device{device}, session{generation::Visuals{.notify = [] { glfwPostEmptyEvent(); },
                              .prepare =
                                  [this](bool is_preview) {
                                      try {
                                          auto& bridge = is_preview ? preview : interop;
                                          if (!bridge) bridge = std::make_shared<Interop>(this->device);
                                      } catch (const vk::DeviceLostError& error) {
                                          this->device.diagnostics.capture(error.what());
                                          throw;
                                      }
                                  },
                              .publish = [this](bool is_preview, const std::uint8_t* pixels, int width, int height, ::cuda::stream_ref stream, std::size_t slot) -> std::shared_ptr<const void> {
                                  try {
                                      auto& bridge     = is_preview ? preview : interop;
                                      const auto bytes = std::size_t(width) * height * 4;
                                      for (const auto& target : bridge->slots)
                                          if (target.buffer.size < bytes && target.value && target.timeline.getCounterValue() < target.value + 1) return {};
                                      bridge->prepare(width, height, stream);
                                      const auto& target = bridge->slots[slot];
                                      if (target.value && target.timeline.getCounterValue() < target.value + 1) return {};
                                      const auto ready = bridge->publish(pixels, width, height, stream, slot);
                                      return std::make_shared<const PresentedFrame>(bridge, slot, ready);
                                  } catch (const vk::DeviceLostError& error) {
                                      this->device.diagnostics.capture(error.what());
                                      throw;
                                  }
                              },
                              .available = [this](std::size_t slot) -> bool {
                                  try {
                                      const auto& target = preview->slots[slot];
                                      return !target.value || target.timeline.getCounterValue() >= target.value + 1;
                                  } catch (const vk::DeviceLostError& error) {
                                      this->device.diagnostics.capture(error.what());
                                      throw;
                                  }
                              },
                              .capture = lan::capture_preview}} {}
} // namespace genesia::editor
