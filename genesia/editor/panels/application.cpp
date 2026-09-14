module;
#include <Windows.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <shellapi.h>
module genesia.editor.panels.application;
import genesia.editor.widgets.controls;
import genesia.editor.widgets.tags;
import genesia.io.files;
import genesia.editor.panels.datasets;
import genesia.runtime.session;
import genesia.runtime.catalog;
import genesia.project;
import genesia.prompt.preset;
import genesia.editor.graphics.bridge;
import std;
namespace genesia::editor {
    Workspace::ControlLayout control_layout(const Workspace& workspace, const float scale, const ImVec2 size, const Workspace::Picture& image) {
        Workspace::ControlLayout layout{};
        layout.different             = workspace.page == Workspace::Page::generation && image.texture && (image.width != workspace.draft.width || image.height != workspace.draft.height);
        const float dimensions_width = workspace.page == Workspace::Page::generation ? 112 * scale + (layout.different ? ImGui::CalcTextSize("Next").x + 12 * scale : 0) : image.texture ? ImGui::CalcTextSize(std::format("{} \xC3\x97 {}", image.width, image.height).c_str()).x : 0;
        if (layout.different) layout.image_label_width = ImGui::CalcTextSize(std::format("{} {} \xC3\x97 {} \xE2\x86\x92", image.preview ? "Preview" : "Image", image.width, image.height).c_str()).x + 12 * scale;
        layout.right_width    = layout.image_label_width + dimensions_width + (image.texture ? 116 * scale : 0);
        const float available = size.x - workspace.dataset_sidebar.width * workspace.dataset_sidebar.amount - 2 * bottom_margin * scale;
        layout.image_above    = layout.right_width > available;
        if (layout.image_above) layout.right_width -= layout.image_label_width;
        return layout;
    }


    void generation_settings(Workspace& workspace, const float scale, const ImVec2 size) {
        const float row_height       = 32 * scale;
        const float seed_label_width = ImGui::CalcTextSize("Seed").x + 12 * scale;
        const float mode_width       = ImGui::CalcTextSize("Random").x + 16 * scale;
        const float panel_width      = std::max(336 * scale, seed_label_width + mode_width + ImGui::CalcTextSize("18446744073709551615").x + 48 * scale);
        ImGui::SetNextWindowPos({size.x - 12 * scale, (top_strip_height + 6) * scale}, ImGuiCond_Always, {1, 0});
        ImGui::SetNextWindowSize({panel_width, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16 * scale, 12 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 12 * scale);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, scale);
        ImGui::PushStyleColor(ImGuiCol_PopupBg, {0.075F, 0.078F, 0.09F, 0.94F});
        ImGui::PushStyleColor(ImGuiCol_Border, {1, 1, 1, 0.055F});
        const bool open = ImGui::BeginPopup("Generation settings", ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(3);
        if (!open) return;
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {24 * scale, 16 * scale});
        const float width = ImGui::GetContentRegionAvail().x;
        const ImVec2 field_size{(width - 24 * scale) / 2, row_height};
        constexpr int steps_step = 1;
        number_field("##Steps", "Steps", ImGuiDataType_S32, &workspace.draft.steps, field_size, &steps_step, "%d", scale, workspace.parameter_edit);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Sampling steps for the next image\nUp / Down: 1\nEnter to confirm, Esc to undo");
        ImGui::SameLine(0, 24 * scale);
        constexpr float cfg_step = 0.1F;
        number_field("##CFG", "CFG", ImGuiDataType_Float, &workspace.draft.cfg, field_size, &cfg_step, "%.1f", scale, workspace.parameter_edit);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Guidance scale for the next image\nUp / Down: 0.1\nEnter to confirm, Esc to undo");
        const auto origin = ImGui::GetCursorScreenPos();
        auto* draw        = ImGui::GetWindowDrawList();
        draw->AddLine({origin.x, origin.y - 8 * scale}, {origin.x + width, origin.y - 8 * scale}, ImGui::GetColorU32(ImVec4{1, 1, 1, 0.06F}), scale);
        draw->AddText({origin.x + 4 * scale, origin.y + (row_height - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "Seed");
        if (!workspace.random_seed) {
            ImGui::SetCursorScreenPos({origin.x + seed_label_width, origin.y});
            number_field("##Seed", "", ImGuiDataType_U64, &workspace.seed, {width - seed_label_width - mode_width - 8 * scale, row_height}, nullptr, "%llu", scale, workspace.parameter_edit);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Seed for the next image\nEnter to confirm, Esc to undo");
        }
        ImGui::SetCursorScreenPos({origin.x + width - mode_width, origin.y});
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6 * scale);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8 * scale, (row_height - ImGui::GetFontSize()) / 2});
        ImGui::PushStyleColor(ImGuiCol_Button, {1, 1, 1, workspace.random_seed ? 0.0F : 0.025F});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {1, 1, 1, 0.06F});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.55F, 0.48F, 0.92F, 0.14F});
        ImGui::PushStyleColor(ImGuiCol_Text, workspace.random_seed ? ImVec4{0.70F, 0.71F, 0.77F, 1} : ImVec4{0.74F, 0.70F, 0.94F, 1});
        const bool mode_clicked = ImGui::Button(workspace.random_seed ? "Random###SeedMode" : "Fixed###SeedMode", {mode_width, row_height});
        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar(2);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\nClick: Random / Fixed", workspace.random_seed ? "A new seed is chosen when queued." : "Reuse this seed for each image.");
        const bool row_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect(origin, {origin.x + width, origin.y + row_height});
        if (mode_clicked || (row_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle))) {
            workspace.commit_parameters();
            workspace.random_seed = !workspace.random_seed;
        }
        if (workspace.repaint.has_value()) {
            ImGui::TextDisabled("Denoise");
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##Denoise", &workspace.denoise, 0, 1, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        }
        if (std::ranges::any_of(workspace.task_status, [](const auto& entry) { return entry.second.kind == runtime::Kind::generate; })) {
            ImGui::Separator();
            ImGui::TextDisabled("IMAGE GENERATION");
            operation_activity(workspace, {runtime::Kind::generate}, "");
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) ImGui::CloseCurrentPopup();
        ImGui::PopStyleVar();
        ImGui::EndPopup();
    }

    void top_strip(Workspace& workspace, const float scale, const ImVec2 size) {
        bool active{}, loaded{true}, failed{}, unavailable{};
        std::optional<std::uint64_t> stopping_image;
        int steps{};
        runtime::Kind active_kind{runtime::Kind::generate};
        double elapsed{};
        if (workspace.runtime) {
            active      = workspace.session_state.active.has_value();
            loaded      = workspace.session_state.generation.model_ready;
            failed      = !workspace.session_state.error.empty();
            unavailable = workspace.session_state.finished;
            if (active) {
                active_kind = workspace.session_state.active->kind;
                steps       = workspace.session_state.generation.steps;
                elapsed     = std::chrono::duration<double>(std::chrono::steady_clock::now() - workspace.session_state.active->started).count();
                if (active_kind == runtime::Kind::generate && workspace.session_state.active->state == runtime::State::running) stopping_image = workspace.session_state.active->id;
            }
        }
        const auto stage         = workspace.session_state.generation.stage;
        const auto completed     = workspace.session_state.generation.completed;
        const bool stopping      = active && workspace.session_state.active->stopping;
        const bool working       = active && !failed;
        const bool indeterminate = working && (active_kind != runtime::Kind::generate || stage != runtime::GenerationStage::sampling || stopping);
        const double now         = glfwGetTime();
        if (working) {
            workspace.progress_alpha = 1;
            if (stopping) workspace.progress_label = "Stopping";
            else if (!loaded) workspace.progress_label = "Loading model";
            else if (stage == runtime::GenerationStage::sampling) workspace.progress_label = std::format("{} / {}", completed, steps);
            else if (stage == runtime::GenerationStage::decoding || stage == runtime::GenerationStage::transferring || stage == runtime::GenerationStage::complete) workspace.progress_label = "Finishing image";
            else workspace.progress_label = "Preparing";
            if (!stopping && active_kind != runtime::Kind::generate) workspace.progress_label = std::array{"Generate", "Training", "Inference", "Audit", "Moving image", "Undo", "Classifying folder", "Assigning type"}[static_cast<int>(active_kind)];
            workspace.progress_time = active ? std::format("{:.1f}s", elapsed) : "";
        } else {
            workspace.progress_alpha = failed ? 0 : std::max(0.0F, workspace.progress_alpha - ImGui::GetIO().DeltaTime / 0.15F);
            if (!failed && (workspace.page == Workspace::Page::generation || workspace.repaint)) {
                bool latest = true;
                for (const auto& [id, task] : std::views::reverse(workspace.task_status)) {
                    if (task.kind != runtime::Kind::generate) continue;
                    const auto state = runtime::states[std::size_t(task.state)];
                    if (state == "saving" || state == "queued" || (latest && state == "failed")) {
                        workspace.progress_label = state == "saving" ? "Saving image" : state == "queued" ? "Image queued" : "Generation failed";
                        workspace.progress_time.clear();
                        workspace.progress_alpha = 1;
                        active_kind              = runtime::Kind::generate;
                        break;
                    }
                    latest = false;
                }
            }
        }
        if (working) workspace.refresh_at = now + (indeterminate ? 1.0 / 30 : 0.1);
        else if (workspace.progress_alpha > 0 && workspace.progress_alpha < 1) workspace.refresh_at = now + 1.0 / 60;

        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({size.x, top_strip_height * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::Begin("##ApplicationStrip", nullptr, overlay | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoNavFocus);
        ImGui::PopStyleVar();
        const auto minimum = ImGui::GetWindowPos();
        const ImVec2 maximum{minimum.x + ImGui::GetWindowWidth(), minimum.y + ImGui::GetWindowHeight()};
        auto* draw = ImGui::GetWindowDrawList();
        if (workspace.progress_alpha > 0) {
            draw->PushClipRect(minimum, maximum, false);
            if (indeterminate) {
                const float width         = maximum.x - minimum.x;
                const float segment_width = std::min(width * 0.18F, 180 * scale);
                const float x             = minimum.x - segment_width + float(std::fmod(now * 180 * scale, double(width + segment_width)));
                draw->AddRectFilled({x, minimum.y}, {x + segment_width, minimum.y + 2 * scale}, ImGui::GetColorU32(ImVec4{0.59F, 0.57F, 0.95F, 0.60F}));
            } else {
                const float fraction = working ? std::min(1.0F, float(completed) / steps) : 1.0F;
                const float x        = std::lerp(minimum.x, maximum.x, fraction);
                draw->AddRectFilled(minimum, {x, minimum.y + 2 * scale}, ImGui::GetColorU32(ImVec4{0.59F, 0.57F, 0.95F, 0.65F * workspace.progress_alpha}));
            }
            draw->PopClipRect();
        }
        ImGui::SetCursorPos({12 * scale, 4 * scale});
        if (workspace.page == Workspace::Page::generation) {
            if (text_button("##Application", "GENESIA", scale)) ImGui::OpenPopup("Application");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Right-click canvas: open Raw\n`: datasets\nTab: Prompt\nF11: fullscreen\nEsc: exit");
        } else {
            if (text_button("##Back", "\xE2\x80\xB9", scale)) workspace.back();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Back one level\nRight-click the canvas to return");
            ImGui::SameLine(0, 4 * scale);
            const auto count  = workspace.collection ? workspace.collection->images.size() : 0;
            const auto number = count ? workspace.current_position().index + 1 : 0;
            const auto label  = std::format("{}  \xC2\xB7  {} / {}", workspace.collection_key, number, count);
            if (text_button("##Location", label.c_str(), scale, std::min(ImGui::CalcTextSize(label.c_str()).x + 24 * scale, size.x * 0.42F))) {
                workspace.dataset_sidebar.open = !workspace.dataset_sidebar.open;
                if (workspace.dataset_sidebar.open) workspace.expand_dataset_roots = true;
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n`: datasets\nTab: Prompt\nLeft-click image: center / inspect\nRight-click canvas: back\nEsc: exit", workspace.collection_key.c_str());
        }
        const float left_end = ImGui::GetItemRectMax().x + 4 * scale;
        if (ImGui::BeginPopup("Application")) {
            ImGui::MenuItem("Live preview", nullptr, &workspace.preview_enabled);
            ImGui::Separator();
            if (ImGui::MenuItem("Save prompt", "Ctrl+S", false, workspace.page == Workspace::Page::generation)) workspace.save_prompt();
            if (ImGui::MenuItem("Open output folder")) ShellExecuteW(workspace.window.native_window, L"open", project::raw.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            ImGui::EndPopup();
        }
        const bool submission     = workspace.page == Workspace::Page::generation || workspace.repaint.has_value();
        const float primary_width = submission ? 150 * scale : 0;
        const float stop_width    = stopping_image ? ImGui::CalcTextSize("Stop image").x + 28 * scale : 0;
        const char* label         = workspace.progress_alpha > 0 ? workspace.progress_label.c_str() : "";
        const float alpha         = workspace.progress_alpha;
        const float right_start   = maximum.x - primary_width - stop_width;
        const float label_width   = *label ? std::max(ImGui::CalcTextSize("Finishing image").x, ImGui::CalcTextSize(label).x) : 0;
        const auto time_text      = ImGui::CalcTextSize(workspace.progress_time.c_str());
        const float time_width    = std::max(ImGui::CalcTextSize("0000.0s").x, time_text.x);
        const bool show_time      = *label && !workspace.progress_time.empty() && label_width + 16 * scale + time_width <= right_start - left_end - 32 * scale;
        const float status_width  = label_width + (show_time ? 16 * scale + time_width : 0);
        const float group_left    = right_start - status_width - (*label ? 12 * scale : 0) - 8 * scale;
        control_shade({group_left, minimum.y + 4 * scale}, maximum, scale);
        if (*label) {
            ImGui::SetCursorScreenPos({group_left + 8 * scale, minimum.y + 4 * scale});
            ImGui::Dummy({status_width, control_height * scale});
            const auto origin = ImGui::GetItemRectMin();
            const auto text   = ImGui::CalcTextSize(label);
            const float y     = origin.y + (control_height * scale - text.y) / 2;
            draw->AddText({origin.x, y}, ImGui::GetColorU32(ImVec4{0.89F, 0.89F, 0.94F, alpha}), label);
            if (show_time) {
                draw->AddCircleFilled({origin.x + label_width + 8 * scale, origin.y + control_height * scale / 2}, 1.25F * scale, ImGui::GetColorU32(ImVec4{0.57F, 0.58F, 0.64F, alpha}));
                draw->AddText({origin.x + status_width - time_text.x, y}, ImGui::GetColorU32(ImVec4{0.67F, 0.68F, 0.74F, alpha}), workspace.progress_time.c_str());
            }
            if (ImGui::IsItemHovered()) {
                if (submission && active_kind == runtime::Kind::generate) {
                    ImGui::SetTooltip("Click: image generation progress and cancellation");
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) ImGui::OpenPopup("Generation settings");
                } else if (!workspace.progress_time.empty()) ImGui::SetTooltip("Elapsed: %s", workspace.progress_time.c_str());
            }
        }
        if (stopping_image) {
            ImGui::SetCursorScreenPos({right_start, minimum.y + 4 * scale});
            ImGui::BeginDisabled(stopping);
            if (text_button("##StopImage", "Stop image", scale)) workspace.runtime->session.cancel(*stopping_image);
            ImGui::EndDisabled();
        }
        if (submission) {
            const bool valid   = workspace.repaint ? workspace.root && workspace.root->ready && workspace.repaints.at(workspace.repaint->source.sha)->editor.valid : workspace.prompt_editor.valid;
            const bool enabled = !unavailable && !ImGui::GetTopMostPopupModal() && valid;
            if (ImGui::IsPopupOpen("Generation settings") && ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect({maximum.x - primary_width, minimum.y}, maximum) && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                workspace.commit_parameters();
                ImGui::ClosePopupsOverWindow(ImGui::GetCurrentWindow(), false);
            }
            ImGui::SetCursorScreenPos({maximum.x - primary_width, minimum.y});
            ImGui::PushFont(nullptr, 18);
            ImGui::BeginDisabled(!enabled);
            const bool clicked = text_button("##Submit", workspace.repaint ? "Repaint" : "Generate", scale, primary_width, false, workspace.repaint ? ImVec4{0.53F, 0.80F, 0.72F, 1} : ImVec4{0.70F, 0.65F, 0.97F, 1});
            ImGui::EndDisabled();
            const bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
            if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                workspace.commit_parameters();
                ImGui::OpenPopup("Generation settings");
            }
            ImGui::PopFont();
            if (hovered && !ImGui::IsPopupOpen("Generation settings")) ImGui::SetTooltip("%s\nCtrl+Shift+`\nRight-click: settings, image progress and cancellation", workspace.repaint ? "Repaint the green source into Raw" : "Generate a new image into Raw");
            generation_settings(workspace, scale, size);
            if (enabled && (clicked || ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_GraveAccent, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive))) workspace.submit();
        }
        workspace.window.drag_region = {};
        if (!ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) workspace.window.drag_region = {left_end, 0, group_left - 4 * scale, maximum.y};
        ImGui::End();
    }

    void bottom_controls(Workspace& workspace, const float scale, const ImVec2 size, const Workspace::ControlLayout& layout, const Workspace::Picture& image) {
        if (workspace.page != Workspace::Page::generation && !image.texture) return;
        const float y     = size.y - (bottom_margin + control_height) * scale;
        const float right = size.x - bottom_margin * scale;
        const float x     = right - layout.right_width;
        ImGui::SetNextWindowPos({x, y});
        ImGui::SetNextWindowSize({layout.right_width, control_height * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::Begin("##ImageControls", nullptr, overlay | ImGuiWindowFlags_NoBackground);
        ImGui::PopStyleVar();
        if (image.texture) control_shade({x, y}, {right, y + control_height * scale}, scale);
        float dimensions_x = x;
        auto* draw         = ImGui::GetWindowDrawList();
        if (layout.different) {
            const float info_x = layout.image_above ? right - layout.image_label_width : dimensions_x;
            const float info_y = layout.image_above ? y - (control_height + 8) * scale : y;
            if (layout.image_above) control_shade({info_x, info_y}, {right, info_y + control_height * scale}, scale);
            draw->PushClipRect({0, 0}, size, false);
            draw->AddText({info_x, info_y + (control_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImVec4{0.54F, 0.55F, 0.61F, 1}), std::format("{} {} \xC3\x97 {} \xE2\x86\x92", image.preview ? "Preview" : "Image", image.width, image.height).c_str());
            draw->PopClipRect();
            if (!layout.image_above) dimensions_x += layout.image_label_width;
        }
        if (workspace.page == Workspace::Page::generation) {
            const ImVec2 dimensions_origin{dimensions_x, y};
            if (layout.different) {
                draw->AddText({dimensions_x + 4 * scale, y + (control_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "Next");
                dimensions_x += ImGui::CalcTextSize("Next").x + 12 * scale;
            }
            ImGui::SetCursorScreenPos({dimensions_x, y});
            constexpr int dimension_step = 64;
            number_field("##Width", "", ImGuiDataType_S32, &workspace.draft.width, {48 * scale, control_height * scale}, &dimension_step, "%d", scale, workspace.parameter_edit);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next image width\nUp / Down: 64 px\nMiddle-click: swap dimensions\nRight-click: presets");
            draw->AddText({dimensions_x + 51 * scale, y + (control_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "\xC3\x97");
            ImGui::SetCursorScreenPos({dimensions_x + 64 * scale, y});
            number_field("##Height", "", ImGuiDataType_S32, &workspace.draft.height, {48 * scale, control_height * scale}, &dimension_step, "%d", scale, workspace.parameter_edit);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next image height\nUp / Down: 64 px\nMiddle-click: swap dimensions\nRight-click: presets");
            const bool dimensions_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect(dimensions_origin, ImGui::GetItemRectMax());
            if (dimensions_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
                workspace.commit_parameters();
                std::swap(workspace.draft.width, workspace.draft.height);
            }
            if (dimensions_hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                workspace.commit_parameters();
                ImGui::OpenPopup("Resolution presets");
            }
            if (ImGui::BeginPopup("Resolution presets")) {
                for (const auto& [label, width, height] : std::array{std::tuple{"Portrait  1024 x 1536", 1024, 1536}, std::tuple{"Square  1024 x 1024", 1024, 1024}, std::tuple{"Landscape  1536 x 1024", 1536, 1024}}) {
                    if (ImGui::MenuItem(label, nullptr, workspace.draft.width == width && workspace.draft.height == height)) {
                        workspace.draft.width  = width;
                        workspace.draft.height = height;
                    }
                }
                if (image.texture) {
                    ImGui::Separator();
                    if (ImGui::MenuItem("Use image size")) {
                        workspace.draft.width  = image.width;
                        workspace.draft.height = image.height;
                    }
                }
                ImGui::EndPopup();
            }
        } else if (image.texture) {
            ImGui::SetCursorScreenPos({dimensions_x, y});
            const auto label = std::format("{} \xC3\x97 {}", image.width, image.height);
            ImGui::Dummy({ImGui::CalcTextSize(label.c_str()).x, control_height * scale});
            draw->AddText({dimensions_x, y + (control_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), label.c_str());
        }
        if (image.texture) {
            ImGui::SetCursorScreenPos({right - 104 * scale, y});
            const ImVec2 dimensions{float(image.width), float(image.height)};
            const float fitted = std::min(workspace.view_available.x / dimensions.x, workspace.view_available.y / dimensions.y);
            if (text_button("##View", std::format("{}{:.0f}%", workspace.view.fit ? "Fit \xC2\xB7 " : "", workspace.view.zoom * 100).c_str(), scale, 104 * scale)) {
                if ((workspace.page == Workspace::Page::dataset || workspace.page == Workspace::Page::audit) && !workspace.repaint) workspace.viewing = Workspace::View::inspect;
                const double now = glfwGetTime();
                workspace.view.scale_to(std::max(1.0F, fitted), {}, dimensions, fitted >= 1, now);
                workspace.animate_until = std::max(workspace.animate_until, now + 0.12);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Left-click: view at 100%%\nMiddle-click: fit image\nFit is the minimum zoom\nImage: scroll to zoom, drag to pan, double-click to toggle fit / 100%%");
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
                    const double now = glfwGetTime();
                    workspace.view.scale_to(fitted, {}, dimensions, true, now);
                    workspace.animate_until = std::max(workspace.animate_until, now + 0.12);
                }
            }
        }
        ImGui::End();
    }
} // namespace genesia::editor
