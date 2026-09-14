module;
#include <GLFW/glfw3.h>
#include <genesia/cuda.h>
module genesia.editor.graphics.bridge;
import std;
namespace genesia::editor {
    WorkspaceRuntime::WorkspaceRuntime(graphics::Device& device)
        : device{device}, session{generation::Visuals{.notify = [] { glfwPostEmptyEvent(); },
                              .prepare =
                                  [this](bool is_preview, int width, int height, ::cuda::stream_ref stream) {
                                      auto& bridge = is_preview ? preview : interop;
                                      if (!bridge) bridge = std::make_shared<Interop>(this->device);
                                  },
                              .publish = [this](bool is_preview, const std::uint8_t* pixels, int width, int height, ::cuda::stream_ref stream, std::size_t slot) -> std::shared_ptr<const void> {
                                  auto& bridge     = is_preview ? preview : interop;
                                  const auto bytes = std::size_t(width) * height * 4;
                                  for (const auto& target : bridge->slots)
                                      if (target.buffer.size < bytes && target.value && target.timeline.getCounterValue() < target.value + 1) return {};
                                  bridge->prepare(width, height, stream);
                                  const auto& target = bridge->slots[slot];
                                  if (target.value && target.timeline.getCounterValue() < target.value + 1) return {};
                                  const auto ready = bridge->publish(pixels, width, height, stream, slot);
                                  return std::make_shared<const PresentedFrame>(bridge, slot, ready);
                              },
                              .available = [this](std::size_t slot) -> bool {
                                  const auto& target = preview->slots[slot];
                                  return !target.value || target.timeline.getCounterValue() >= target.value + 1;
                              }}} {}
} // namespace genesia::editor
