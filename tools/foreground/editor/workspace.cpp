module;
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
module foreground.editor.workspace;
import tools.files;
import tools.editor.style;
import std;
namespace foreground::editor {
    Workspace::Workspace(tools::editor::WindowPlatform& platform, const float& scale) : window{platform}, dpi{scale}, session{[] { glfwPostEmptyEvent(); }} {}
    void Workspace::receive() {
        auto delivery       = session.drain();
        const auto previous = state.revision;
        state               = std::move(delivery.state);
        if (state.revision != previous && !state.result.error.empty()) error = state.result.error;
    }
    void Workspace::drop() {
        if (!window.drop_error.empty()) error = std::exchange(window.drop_error, {});
        auto paths = std::exchange(window.dropped, {});
        if (paths.empty()) return;
        try {
            if (paths.size() != 1) throw std::runtime_error{"Drop one folder"};
            session.submit({paths.front()});
            state = session.drain().state;
            error = state.result.error;
        } catch (const std::exception& failure) {
            error = failure.what();
        }
        window.redraw = true;
    }
    void Workspace::draw() {
        constexpr auto title = "Generate Foreground Masks";
        const auto& style    = ImGui::GetStyle();
        float width          = 360 * dpi;
        if (state.state == runtime::State::idle) {
            ImGui::PushFont(nullptr, 30);
            width = std::max(width, ImGui::CalcTextSize(title).x + 2 * style.WindowPadding.x);
            ImGui::PopFont();
        }
        width = std::min(width, float(window.extent_limit.width));
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({width, float(window.extent_limit.height)});
        ImGui::Begin("Foreground", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
        if (state.state == runtime::State::idle) tools::editor::draw_welcome(title, "Drag in an image folder");
        else if (!state.busy) {
            const auto origin     = ImGui::GetCursorPos();
            constexpr auto label  = "Drag a folder";
            const auto dimensions = ImGui::CalcTextSize(label);
            ImGui::SetCursorPos({(width - dimensions.x) / 2, origin.y + (64 * dpi - dimensions.y) / 2});
            ImGui::TextUnformatted(label);
            ImGui::SetCursorPos(origin);
            ImGui::Dummy({ImGui::GetContentRegionAvail().x, 64 * dpi});
        } else {
            const auto label = state.stopping ? "Stopping..." : state.progress.stage == Stage::scanning ? "Scanning..." : state.progress.stage == Stage::loading_model ? "Loading model..." : "Generating masks";
            ImGui::TextUnformatted(label);
            ImGui::SameLine();
            ImGui::TextDisabled("%zu / %zu", state.progress.completed, state.progress.total);
            const float ratio = state.progress.total ? float(state.progress.completed) / float(state.progress.total) : 0;
            ImGui::ProgressBar(ratio, {-1, 4 * dpi}, "");
            if (!state.progress.file.empty()) {
                ImGui::TextWrapped("%s", tools::files::utf8(state.progress.file.filename()).c_str());
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tools::files::utf8(state.progress.file).c_str());
            }
            ImGui::BeginDisabled(state.stopping);
            if (ImGui::Button("Stop")) session.cancel();
            ImGui::EndDisabled();
        }
        if (!state.busy && state.state != runtime::State::idle) {
            ImGui::Separator();
            const auto label = state.state == runtime::State::complete ? "Complete" : state.state == runtime::State::stopped ? "Stopped" : "Failed";
            ImGui::Text("%s  /  %zu of %zu saved", label, state.result.completed, state.result.total);
            ImGui::PushTextWrapPos();
            ImGui::TextDisabled("%s", tools::files::utf8(state.result.output).c_str());
            ImGui::PopTextWrapPos();
        }
        if (!error.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, {0.95F, 0.59F, 0.55F, 1});
            ImGui::TextWrapped("%s", error.c_str());
            ImGui::PopStyleColor();
            if (ImGui::SmallButton("Dismiss")) error.clear();
        }
        const auto* layout           = ImGui::GetCurrentWindow();
        const float height           = layout->DC.CursorMaxPos.y - layout->DC.CursorStartPos.y + 2 * style.WindowPadding.y;
        window.content_extent.width  = static_cast<std::uint32_t>(std::ceil(width));
        window.content_extent.height = static_cast<std::uint32_t>(std::ceil(std::min(height, float(window.extent_limit.height))));
        if (!state.busy && !window.dragged.empty()) {
            const auto size = ImGui::GetIO().DisplaySize;
            ImGui::GetForegroundDrawList()->AddRect({5 * dpi, 5 * dpi}, {size.x - 5 * dpi, size.y - 5 * dpi}, ImGui::GetColorU32(ImGuiCol_PlotHistogram), 10 * dpi, 0, 2 * dpi);
        }
        window.drag_requested = !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive() && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
        ImGui::End();
    }
} // namespace foreground::editor
