module;
#include <Windows.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
#include <shlobj.h>
#include <nlohmann/json.hpp>
module edit.editor.workspace;
import tools.files;
import tools.editor.style;
import std;
namespace edit::editor {
    Workspace::Workspace(tools::editor::WindowPlatform& platform, const float& scale) : window{platform}, dpi{scale}, session{[] { glfwPostEmptyEvent(); }} {
        PWSTR directory{};
        const auto status = SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &directory);
        if (FAILED(status)) throw std::system_error{static_cast<int>(status), std::system_category(), "Find Edit settings directory"};
        settings = std::filesystem::path{directory} / "Genesia" / "Edit" / "settings.json";
        CoTaskMemFree(directory);
        if (std::filesystem::exists(settings)) prompt = tools::files::read_json(settings).at("prompt").get<std::string>();
    }
    void Workspace::receive() {
        const auto now = std::chrono::steady_clock::now();
        if (connection_check.valid() && connection_check.wait_for(std::chrono::seconds{0}) == std::future_status::ready) {
            connected             = connection_check.get();
            next_connection_check = now + std::chrono::seconds{3};
            window.redraw         = true;
        }
        if (!connection_check.valid() && now >= next_connection_check) connection_check = std::async(std::launch::async, check_connection);
        auto delivery       = session.drain();
        const auto previous = state.revision;
        state               = std::move(delivery.state);
        if (state.revision != previous && !state.result.error.empty()) error = state.result.error;
        if (save_at && std::chrono::steady_clock::now() >= *save_at) {
            try {
                save_prompt();
            } catch (const std::exception& failure) {
                error         = failure.what();
                window.redraw = true;
            }
        }
    }
    void Workspace::drop() {
        if (!window.drop_error.empty()) error = std::exchange(window.drop_error, {});
        auto paths = std::exchange(window.dropped, {});
        if (paths.empty()) return;
        try {
            if (state.busy) throw std::runtime_error{"Stop the current batch before dropping another input"};
            if (paths.size() != 1) throw std::runtime_error{"Drop one PNG image or one image folder"};
            if (prompt.find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error{"Enter a positive prompt before dropping images"};
            save_prompt();
            session.submit({paths.front(), prompt});
            state = session.drain().state;
            error = state.result.error;
        } catch (const std::exception& failure) {
            error = failure.what();
        }
        window.redraw = true;
    }
    void Workspace::draw() {
        const float width = std::min(420 * dpi, float(window.extent_limit.width));
        const auto& style = ImGui::GetStyle();
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({width, float(window.extent_limit.height)});
        ImGui::Begin("Edit", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
        if (state.state == runtime::State::idle) tools::editor::draw_welcome("Edit Images with Qwen", "Write a prompt, then drop an image or folder");
        ImGui::TextDisabled("Positive prompt");
        const auto connection_label = !connected ? "ComfyUI: Checking..." : *connected ? "ComfyUI: Connected" : "ComfyUI: Offline";
        const auto connection_color = !connected ? style.Colors[ImGuiCol_TextDisabled] : *connected ? style.Colors[ImGuiCol_PlotHistogram] : ImVec4{0.95F, 0.59F, 0.55F, 1};
        ImGui::SameLine(width - style.WindowPadding.x - ImGui::CalcTextSize(connection_label).x);
        ImGui::TextColored(connection_color, "%s", connection_label);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", server_url.data());
        const float text_width    = ImGui::GetContentRegionAvail().x - 2 * style.FramePadding.x - style.ScrollbarSize;
        const float text_height   = ImGui::CalcTextSize(prompt.c_str(), nullptr, false, text_width).y;
        const float prompt_height = std::clamp(text_height, 3 * ImGui::GetTextLineHeight(), 8 * ImGui::GetTextLineHeight()) + 2 * style.FramePadding.y;
        ImGui::BeginDisabled(state.busy);
        if (ImGui::InputTextMultiline("##prompt", &prompt, {-1, prompt_height}, ImGuiInputTextFlags_WordWrap)) {
            save_at       = std::chrono::steady_clock::now() + std::chrono::milliseconds{300};
            window.redraw = true;
        }
        ImGui::EndDisabled();
        if (!state.busy && state.state != runtime::State::idle) {
            const auto origin     = ImGui::GetCursorPos();
            constexpr auto label  = "Drag a PNG or folder";
            const auto dimensions = ImGui::CalcTextSize(label);
            ImGui::SetCursorPos({(width - dimensions.x) / 2, origin.y + (64 * dpi - dimensions.y) / 2});
            ImGui::TextUnformatted(label);
            ImGui::SetCursorPos(origin);
            ImGui::Dummy({ImGui::GetContentRegionAvail().x, 64 * dpi});
        } else if (state.busy) {
            ImGui::TextUnformatted(state.stopping || state.progress.stage == Stage::stopping ? "Stopping..." : state.progress.stage == Stage::preparing ? "Preparing..." : "Editing");
            ImGui::SameLine();
            ImGui::TextDisabled("%zu / %zu", state.progress.completed, state.progress.total);
            const float ratio = state.progress.total ? float(state.progress.completed) / float(state.progress.total) : 0;
            ImGui::ProgressBar(ratio, {-1, 4 * dpi}, "");
            if (!state.progress.file.empty()) ImGui::TextWrapped("%s", tools::files::utf8(state.progress.file.filename()).c_str());
            ImGui::BeginDisabled(state.stopping || state.progress.stage == Stage::stopping);
            if (ImGui::Button("Stop")) session.cancel();
            ImGui::EndDisabled();
        }
        if (!state.busy && state.state != runtime::State::idle) {
            ImGui::Separator();
            const auto label = state.state == runtime::State::complete ? "Complete" : state.state == runtime::State::stopped ? "Stopped" : "Failed";
            ImGui::Text("%s  /  %zu of %zu saved", label, state.result.completed, state.result.total);
            if (state.result.completed) {
                ImGui::PushTextWrapPos();
                ImGui::TextDisabled("%s", tools::files::utf8(state.result.output).c_str());
                ImGui::PopTextWrapPos();
            }
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
    void Workspace::save_prompt() {
        save_at.reset();
        tools::files::write_json(settings, {{"prompt", prompt}});
    }
} // namespace edit::editor
