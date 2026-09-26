module;
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
module genesia.editor.viewing.canvas;
import genesia.editor.widgets.controls;
import genesia.editor.widgets.tags;
import genesia.io.files;
import genesia.runtime.session;
import genesia.project;
import genesia.editor.graphics.bridge;
import genesia.editor.viewing.camera;
import std;
namespace genesia::editor {
    bool image_panel(Workspace& workspace, const char* id, const Workspace::Picture& image, const ImVec2 origin, const ImVec2 size, const bool interactive, const float brightness) {
        if (!image.width || !image.height) return false;
        auto* draw = ImGui::GetWindowDrawList();
        ImGui::SetCursorScreenPos(origin);
        ImGui::InvisibleButton(id, size, ImGuiButtonFlags_MouseButtonLeft);
        const bool hovered = ImGui::IsItemHovered();
        const auto& io     = ImGui::GetIO();
        const ImVec2 available{size.x * io.DisplayFramebufferScale.x, size.y * io.DisplayFramebufferScale.y};
        const ImVec2 dimensions{float(image.width), float(image.height)};
        const float fitted = std::min(available.x / dimensions.x, available.y / dimensions.y);
        const double now   = workspace.frame_time;
        if (interactive || (image.file && workspace.page == Workspace::Page::history && *image.file == workspace.position.selected)) {
            workspace.view_available = available;
            workspace.view.update(available, dimensions, now);
        }
        const float zoom    = interactive ? workspace.view.zoom : fitted;
        const ImVec2 center = interactive ? workspace.view.center : ImVec2{0.5F, 0.5F};
        const ImVec2 extent{dimensions.x * zoom / io.DisplayFramebufferScale.x, dimensions.y * zoom / io.DisplayFramebufferScale.y};
        ImVec2 minimum{origin.x + size.x / 2 - center.x * extent.x, origin.y + size.y / 2 - center.y * extent.y};
        const bool over_image = hovered && ImGui::IsMouseHoveringRect(minimum, {minimum.x + extent.x, minimum.y + extent.y});
        if (interactive && image.texture) {
            const bool movable = extent.x > size.x + 1 || extent.y > size.y + 1;
            if (over_image && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && movable && !ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                workspace.view.dragging = true;
                workspace.view.started  = -1;
            }
            if (workspace.view.dragging && ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                workspace.view.center.x -= io.MouseDelta.x * io.DisplayFramebufferScale.x / (dimensions.x * workspace.view.zoom);
                workspace.view.center.y -= io.MouseDelta.y * io.DisplayFramebufferScale.y / (dimensions.y * workspace.view.zoom);
                workspace.view.constrain(available, dimensions);
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) workspace.view.dragging = false;
            if (hovered && io.MouseWheel && !workspace.view.dragging) {
                ImGui::SetKeyOwner(ImGuiKey_MouseWheelY, ImGui::GetItemID());
                const float next = std::clamp(workspace.view.target_zoom * std::pow(1.15F, io.MouseWheel), fitted, std::max(16.0F, fitted));
                const ImVec2 pivot{(io.MousePos.x - origin.x - size.x / 2) * io.DisplayFramebufferScale.x, (io.MousePos.y - origin.y - size.y / 2) * io.DisplayFramebufferScale.y};
                workspace.view.scale_to(next, pivot, dimensions, next <= fitted, now);
            }
            if (over_image && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) workspace.view.scale_to(workspace.view.fit ? std::max(1.0F, fitted) : fitted, {}, dimensions, !workspace.view.fit || fitted >= 1, now);
            if (over_image && movable) workspace.renderer.hand_cursor = workspace.view.dragging ? 1 : 0;
            minimum = {origin.x + size.x / 2 - workspace.view.center.x * extent.x, origin.y + size.y / 2 - workspace.view.center.y * extent.y};
        }
        const bool clicked = image.texture && over_image && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !ImGui::IsMouseDragPastThreshold(ImGuiMouseButton_Left) && io.MouseClickedLastCount[ImGuiMouseButton_Left] == 1;
        if (zoom == 1) {
            minimum.x = std::round(minimum.x * io.DisplayFramebufferScale.x) / io.DisplayFramebufferScale.x;
            minimum.y = std::round(minimum.y * io.DisplayFramebufferScale.y) / io.DisplayFramebufferScale.y;
        }
        const ImVec2 maximum{minimum.x + extent.x, minimum.y + extent.y};
        draw->PushClipRect(origin, {origin.x + size.x, origin.y + size.y}, true);
        if (image.texture) {
            draw->AddImage(image.texture, minimum, maximum, {0, 0}, {1, 1}, ImGui::GetColorU32(ImVec4{brightness, brightness, brightness, 1}));
        } else {
            const char* label = "Loading image";
            if (image.file) {
                const auto cached = workspace.textures.entries.find(*image.file);
                if (cached != workspace.textures.entries.end() && !cached->second.error.empty()) label = "Image read error";
            }
            const auto text = ImGui::CalcTextSize(label);
            draw->AddText({origin.x + (size.x - text.x) / 2, origin.y + (size.y - text.y) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
        }
        draw->PopClipRect();
        return clicked;
    }

    void canvas(Workspace& workspace, const float scale, const ImVec2 size) {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::Begin("##Canvas", nullptr, overlay | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::PopStyleVar();
        constexpr ImVec2 origin{};
        const auto available = workspace.canvas_size;
        // Navigate on press so releasing a popup-dismissal click cannot also change the view.
        const bool navigating = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(origin, {origin.x + available.x, origin.y + available.y}) && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) && ImGui::IsMouseClicked(ImGuiMouseButton_Right);
        ImGui::PushClipRect(origin, {origin.x + available.x, origin.y + available.y}, true);
        std::vector<std::filesystem::path> wanted;
        if (workspace.page == Workspace::Page::generation) {
            if (workspace.generation.saved) wanted.push_back(*workspace.generation.saved);
            const auto image = workspace.resolve_generation();
            if (image.texture) image_panel(workspace, "##GeneratedImage", image, origin, available, true);
            else {
                auto* draw = ImGui::GetWindowDrawList();
                const ImVec2 center{origin.x + available.x / 2, origin.y + available.y / 2};
                const char* title = "Imagine something new.";
                ImGui::PushFont(nullptr, 30);
                const auto text = ImGui::CalcTextSize(title);
                draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(), {center.x - text.x / 2, center.y - 30 * scale}, IM_COL32(218, 219, 230, 255), title);
                ImGui::PopFont();
                const char* subtitle = "A few words. A world of possibilities.";
                const auto hint      = ImGui::CalcTextSize(subtitle);
                draw->AddText({center.x - hint.x / 2, center.y + 20 * scale}, IM_COL32(119, 121, 137, 255), subtitle);
                draw->AddCircle({center.x, center.y - 88 * scale}, 14 * scale, IM_COL32(145, 142, 225, 180), 32, 1.5F * scale);
                draw->AddCircleFilled({center.x + 14 * scale, center.y - 100 * scale}, 3 * scale, IM_COL32(184, 182, 250, 255));
            }
        } else if (!workspace.textures.history.empty()) {
            auto& position = workspace.position;
            if (workspace.viewing == Workspace::View::inspect) {
                const auto& file = workspace.textures.history[position.index];
                wanted.push_back(file);
                const auto image = workspace.resolve_image(file);
                image_panel(workspace, "##InspectedImage", image, origin, available, true);
            } else {
                wanted.push_back(workspace.textures.history[position.index]);
                const auto& io = ImGui::GetIO();
                if (ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(origin, {origin.x + available.x, origin.y + available.y}) && io.MouseWheel) {
                    const auto index = std::clamp<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(position.index) - (io.MouseWheel > 0 ? 1 : -1), 0, workspace.textures.history.size() - 1);
                    workspace.center_image(static_cast<std::size_t>(index));
                    ImGui::SetKeyOwner(ImGuiKey_MouseWheelY, ImGui::GetID("##HistoryScroll"));
                }
                position.scroll = std::lerp(position.scroll, static_cast<float>(position.index), std::min(1.0F, io.DeltaTime / 0.055F));
                if (std::abs(position.scroll - position.index) < 0.001F) position.scroll = static_cast<float>(position.index);
                else workspace.animate_until = glfwGetTime() + 0.1;
                const auto anchor = static_cast<std::size_t>(std::floor(position.scroll));
                const auto first  = anchor > 3 ? anchor - 3 : 0;
                const auto last   = std::min(workspace.textures.history.size(), anchor + 5);
                const float gap   = 24 * scale;
                const auto width  = [&](const std::size_t index) {
                    const auto image = workspace.resolve_image(workspace.textures.history[index]);
                    return std::min(available.x * 0.68F, available.y * image.width / image.height);
                };
                const float advance = anchor + 1 < workspace.textures.history.size() ? (width(anchor) + width(anchor + 1)) / 2 + gap : 0;
                float x             = origin.x + available.x / 2 - (position.scroll - anchor) * advance;
                for (auto i = anchor; i > first; --i) x -= (width(i) + width(i - 1)) / 2 + gap;
                for (auto i = first; i < last; ++i) {
                    const auto& file = workspace.textures.history[i];
                    wanted.push_back(file);
                    const float image_width = width(i);
                    const float brightness  = std::lerp(1.0F, 0.60F, std::min(1.0F, std::abs(float(i) - position.scroll)));
                    ImGui::PushID(static_cast<int>(i));
                    const bool clicked = image_panel(workspace, "##HistoryImage", workspace.resolve_image(file), {x - image_width / 2, origin.y}, {image_width, available.y}, false, brightness);
                    ImGui::PopID();
                    if (clicked) {
                        if (i == position.index && position.scroll == float(position.index)) {
                            workspace.viewing = Workspace::View::inspect;
                            workspace.view    = {};
                        } else workspace.center_image(i);
                    }
                    if (i + 1 < last) x += (image_width + width(i + 1)) / 2 + gap;
                }
            }
        }
        if (workspace.page == Workspace::Page::history && workspace.textures.history.empty()) {
            const char* label = workspace.textures.history_ready ? "No images in Raw" : "Loading Raw history";
            const auto text   = ImGui::CalcTextSize(label);
            ImGui::GetWindowDrawList()->AddText({origin.x + (available.x - text.x) / 2, origin.y + (available.y - text.y) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
        }
        for (const auto& file : wanted) {
            const auto cached = workspace.textures.entries.find(file);
            if (cached != workspace.textures.entries.end() && !cached->second.error.empty()) workspace.action_error = cached->second.error;
        }
        if (workspace.page == Workspace::Page::history && !workspace.textures.history.empty()) {
            const auto selected = std::ranges::find(wanted, workspace.textures.history[workspace.position.index]);
            if (selected != wanted.end()) std::rotate(wanted.begin(), selected, selected + 1);
        }
        if (workspace.generation.saved && !std::ranges::contains(wanted, *workspace.generation.saved)) wanted.push_back(*workspace.generation.saved);
        if (workspace.pending_delete) std::erase(wanted, *workspace.pending_delete);
        workspace.textures.request(std::move(wanted));
        ImGui::PopClipRect();
        ImGui::End();
        if (navigating) {
            if (workspace.page == Workspace::Page::generation) workspace.open_history();
            else workspace.back();
        }
    }
} // namespace genesia::editor
