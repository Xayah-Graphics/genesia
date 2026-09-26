module;
#include <GLFW/glfw3.h>
#include <imgui.h>
module classifier.editor;
import tools.editor.platform.window;
import tools.editor.graphics.renderer;
import classifier.editor.workspace;
import std;
namespace classifier::editor {
    int run(const std::span<const std::string_view> arguments) {
        if (!arguments.empty()) throw std::runtime_error{"The Editor accepts files by drag-and-drop. Use --headless for commands."};
        tools::editor::WindowPlatform window{"Classifier", {280, 136}};
        tools::editor::Renderer renderer{window};
        Workspace workspace{window, renderer.dpi};
        bool closing{};
        std::uint64_t revision{};
        int redraws{2};
        for (;;) {
            glfwPollEvents();
            workspace.receive();
            if (window.take_close_request()) {
                closing = true;
                workspace.session.cancel();
            }
            if (closing && !workspace.state.busy) break;
            if (!closing) workspace.drop();
            if (std::exchange(window.redraw, false) || revision != workspace.state.revision) redraws = 2;
            revision = workspace.state.revision;
            if (!redraws && !ImGui::IsAnyItemActive()) {
                glfwWaitEventsTimeout(1);
                continue;
            }
            if (redraws) --redraws;
            if (!renderer.begin()) continue;
            workspace.draw();
            renderer.present();
            window.resize(window.content_extent);
            if (!glfwGetWindowAttrib(window.window, GLFW_VISIBLE)) glfwShowWindow(window.window);
            if (std::exchange(window.drag_requested, false)) window.move();
            glfwWaitEventsTimeout(workspace.state.busy ? 1.0 / 30 : 1.0 / 60);
        }
        return 0;
    }
} // namespace classifier::editor
