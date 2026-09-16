module;
#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
module genesia.editor.panels.sidebars;
import genesia.editor.widgets.controls;
import genesia.editor.widgets.tags;
import genesia.editor.panels.datasets;
import genesia.prompt.library;
import std;
namespace genesia::editor {
    void preset_dialogs(Workspace& workspace, const float scale) {
        if (!workspace.pending_preset.empty() && !ImGui::IsPopupOpen("Unsaved prompt")) {
            if (!workspace.prompt_editor.commit(workspace.prompt.free, *workspace.catalog)) workspace.pending_preset.clear();
            else if (workspace.prompt == workspace.preset.recipe) workspace.switch_preset();
            else ImGui::OpenPopup("Unsaved prompt");
        }
        ImGui::SetNextWindowSize({420 * scale, 0});
        if (ImGui::BeginPopupModal("Unsaved prompt", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextWrapped("Save changes to %s before switching?", workspace.preset.name.c_str());
            if (!workspace.preset_error.empty()) ImGui::TextWrapped("%s", workspace.preset_error.c_str());
            ImGui::Spacing();
            if (ImGui::Button("Save", {108 * scale, 0}) && workspace.save_prompt()) {
                workspace.switch_preset();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Discard", {108 * scale, 0})) {
                workspace.switch_preset();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", {108 * scale, 0}) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                workspace.pending_preset.clear();
                workspace.preset_error.clear();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (std::exchange(workspace.save_as_requested, false) && workspace.prompt_editor.commit(workspace.prompt.free, *workspace.catalog)) {
            workspace.preset_error.clear();
            workspace.new_preset_name.fill(0);
            ImGui::OpenPopup("Save prompt as");
        }
        ImGui::SetNextWindowSize({420 * scale, 0});
        if (ImGui::BeginPopupModal("Save prompt as", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextDisabled("Preset name");
            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            ImGui::SetNextItemWidth(-1);
            const bool enter = ImGui::InputText("##PresetName", workspace.new_preset_name.data(), workspace.new_preset_name.size(), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCharFilter, [](ImGuiInputTextCallbackData* data) {
                const auto c = data->EventChar;
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' ? 0 : 1;
            });
            ImGui::TextDisabled("Letters, numbers, hyphens and underscores");
            if (!workspace.preset_error.empty()) ImGui::TextWrapped("%s", workspace.preset_error.c_str());
            ImGui::Spacing();
            ImGui::BeginDisabled(workspace.new_preset_name[0] == 0);
            if ((ImGui::Button("Save", {108 * scale, 0}) || enter) && workspace.new_preset_name[0]) {
                try {
                    prompts::Preset next{workspace.new_preset_name.data(), workspace.prompt};
                    prompts::write_preset(workspace.prompt_library->directory, next, *workspace.catalog, false);
                    workspace.preset = std::move(next);
                    workspace.preset_error.clear();
                    ImGui::CloseCurrentPopup();
                } catch (const std::exception& failure) {
                    workspace.preset_error = failure.what();
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Cancel", {108 * scale, 0}) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                workspace.preset_error.clear();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (std::exchange(workspace.final_prompt_requested, false) && workspace.prompt_editor.commit(workspace.prompt.free, *workspace.catalog)) {
            workspace.prompt_panel.update(workspace.prompt, *workspace.catalog);
            if (workspace.prompt_panel.composition.error.empty()) ImGui::OpenPopup("Final prompt");
            else workspace.preset_error = workspace.prompt_panel.error;
        }
        ImGui::SetNextWindowSize({720 * scale, 520 * scale}, ImGuiCond_Appearing);
        if (ImGui::BeginPopupModal("Final prompt", nullptr, ImGuiWindowFlags_NoSavedSettings)) {
            const auto& text = workspace.prompt_panel.composition.text;
            if (ImGui::BeginChild("##FinalText", {0, -40 * scale})) {
                for (std::size_t side = 0; side < 2; ++side) {
                    ImGui::PushID(static_cast<int>(side));
                    ImGui::SeparatorText(side ? "Negative" : "Positive");
                    if (ImGui::SmallButton("Copy")) ImGui::SetClipboardText(text[side].c_str());
                    ImGui::TextWrapped("%s", text[side].c_str());
                    ImGui::PopID();
                }
            }
            ImGui::EndChild();
            if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
        if (!workspace.preset_error.empty() && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) ImGui::OpenPopup("Prompt preset error");
        ImGui::SetNextWindowSize({420 * scale, 0});
        if (ImGui::BeginPopupModal("Prompt preset error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextWrapped("%s", workspace.preset_error.c_str());
            ImGui::Spacing();
            if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                workspace.preset_error.clear();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    void sidebar(Workspace& workspace, const bool left, const float scale, const ImVec2 size, const Workspace::Picture& image) {
        const auto& panel       = left ? workspace.dataset_sidebar : workspace.prompt_sidebar;
        const bool preview_drop = !left && workspace.prompt_panel.incoming.has_value();
        const bool temporary    = preview_drop && workspace.page != Workspace::Page::generation;
        const bool open         = panel.open || preview_drop;
        if (!left && (!open || workspace.page != Workspace::Page::generation)) workspace.prompt_editor.suspend();
        if (panel.amount == 0) return;
        const float visible = panel.width * panel.amount;
        const float top     = (top_strip_height + 24) * scale;
        ImGui::SetNextWindowPos({left ? visible - panel.width : size.x - visible, top});
        ImGui::SetNextWindowSize({panel.width, std::max(1.0F, size.y - top - (bottom_margin + control_height + 16) * scale)});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16 * scale, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, panel.amount);
        if (left) {
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8 * scale, 5 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8 * scale, 6 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4 * scale);
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0);
            ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 5 * scale);
            ImGui::PushStyleColor(ImGuiCol_Header, {0.32F, 0.60F, 0.65F, 0.18F});
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {0.55F, 0.65F, 0.69F, 0.12F});
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, {0.32F, 0.60F, 0.65F, 0.25F});
            ImGui::PushStyleColor(ImGuiCol_Button, {0.18F, 0.21F, 0.24F, 0.65F});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.24F, 0.30F, 0.33F, 0.75F});
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.28F, 0.37F, 0.40F, 0.85F});
            ImGui::PushStyleColor(ImGuiCol_PlotLines, {0.44F, 0.80F, 0.87F, 0.85F});
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, {0.44F, 0.80F, 0.87F, 0.75F});
            ImGui::PushStyleColor(ImGuiCol_Separator, {0.70F, 0.72F, 0.85F, 0.12F});
        }
        std::optional<std::string> selected;
        if (ImGui::Begin(left ? "##DatasetSidebar" : "##PromptSidebar", nullptr, overlay | ImGuiWindowFlags_NoBackground | (open ? ImGuiWindowFlags_None : ImGuiWindowFlags_NoInputs))) {
            ImGui::PushFont(nullptr, 12);
            const float content_y = ImGui::GetCursorPosY() + ImGui::GetFontSize() + 12 * scale;
            if (left) ImGui::TextDisabled("DATASETS");
            else if (temporary) ImGui::TextDisabled("SET PREVIEW");
            else if (workspace.page == Workspace::Page::generation) {
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 0});
                ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, {0, 0.5F});
                ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                if (ImGui::Button(workspace.preset.name.c_str(), {ImGui::CalcTextSize(workspace.preset.name.c_str()).x + 16 * scale, 0})) {
                    try {
                        workspace.preset_names = prompts::list_presets(workspace.prompt_library->directory);
                        ImGui::OpenPopup("Prompt presets");
                    } catch (const std::exception& failure) {
                        workspace.preset_error = failure.what();
                    }
                }
                const auto minimum = ImGui::GetItemRectMin();
                const auto maximum = ImGui::GetItemRectMax();
                const ImVec2 arrow{maximum.x - 5 * scale, (minimum.y + maximum.y) / 2};
                ImGui::GetWindowDrawList()->AddTriangleFilled({arrow.x - 3 * scale, arrow.y - 1.5F * scale}, {arrow.x + 3 * scale, arrow.y - 1.5F * scale}, {arrow.x, arrow.y + 1.5F * scale}, ImGui::GetColorU32(ImGuiCol_Text));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Choose a prompt preset\nCtrl+S: save prompt");
                ImGui::PopStyleColor(2);
                ImGui::PopStyleVar(2);
                if (workspace.prompt != workspace.preset.recipe) {
                    ImGui::SameLine(0, 8 * scale);
                    const auto point = ImGui::GetCursorScreenPos();
                    ImGui::Dummy({8 * scale, ImGui::GetTextLineHeight()});
                    ImGui::GetWindowDrawList()->AddCircleFilled({point.x + 3 * scale, point.y + ImGui::GetTextLineHeight() / 2}, 2 * scale, ImGui::GetColorU32(ImGuiCol_CheckMark));
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Unsaved prompt changes");
                }
                if (ImGui::BeginPopup("Prompt presets")) {
                    for (const auto& name : workspace.preset_names) {
                        if (ImGui::Selectable(name.c_str(), name == workspace.preset.name) && name != workspace.preset.name) {
                            workspace.pending_preset = name;
                            workspace.preset_error.clear();
                        }
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem("Save as...")) workspace.save_as_requested = true;
                    ImGui::EndPopup();
                }
            } else if (workspace.repaint) {
                ImGui::TextDisabled("Repaint edits");
                if (ImGui::BeginPopupContextItem("Repaint edits")) {
                    if (ImGui::MenuItem("Reset changes")) {
                        ImGui::ClearActiveID();
                        workspace.repaints[workspace.repaint->source.sha] = std::make_unique<Workspace::RepaintDraft>(*workspace.textures.entries.at(workspace.repaint->source.sha).record);
                    }
                    ImGui::EndPopup();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Temporary draft for the green source\nRight-click: Reset changes");
            } else ImGui::TextDisabled("Image prompt");
            ImGui::PopFont();
            ImGui::SetCursorPosY(content_y);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            if (left) dataset_controls(workspace, scale);
            else workspace.prompt_panel.status();
            if (ImGui::BeginChild(temporary ? "##PreviewDropContent" : "##SidebarContent", {0, 0}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
                if (left) selected = dataset_contents(workspace);
                else if (workspace.page == Workspace::Page::generation || temporary) {
                    workspace.prompt_panel.draw(workspace.prompt, *workspace.catalog, scale);
                    if (!temporary) workspace.prompt_editor.draw(workspace.prompt.free, workspace.tag_search, *workspace.catalog, scale);
                } else if (workspace.repaint) {
                    auto& edits = *workspace.repaints.at(workspace.repaint->source.sha);
                    ImGui::PushID(workspace.repaint->source.sha.c_str());
                    if (ImGui::Button("Apply current prompt recipe") && workspace.prepare_prompt()) {
                        ImGui::ClearActiveID();
                        edits.text = workspace.prompt_panel.composition.text;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Reset")) {
                        ImGui::ClearActiveID();
                        edits = Workspace::RepaintDraft{*workspace.textures.entries.at(workspace.repaint->source.sha).record};
                    }
                    for (std::size_t side = 0; side < 2; ++side) {
                        ImGui::PushID(static_cast<int>(side));
                        ImGui::SeparatorText(side ? "Negative" : "Positive");
                        ImGui::InputTextMultiline("##Prompt", &edits.text[side], {-1, 230 * scale}, ImGuiInputTextFlags_WordWrap);
                        ImGui::PopID();
                    }
                    ImGui::PopID();
                } else if (image.record) {
                    ImGui::SeparatorText("Positive");
                    ImGui::TextWrapped("%s", image.record->parameters.positive.c_str());
                    ImGui::SeparatorText("Negative");
                    ImGui::TextWrapped("%s", image.record->parameters.negative.c_str());
                } else if (!image.record_error.empty()) ImGui::TextWrapped("%.*s", static_cast<int>(image.record_error.size()), image.record_error.data());
                else ImGui::TextDisabled(image.file ? "Loading image..." : "No image selected.");
                if (preview_drop) {
                    const auto* content = ImGui::GetCurrentWindow();
                    const ImVec2 pointer{workspace.window.drop_position[0], workspace.window.drop_position[1]};
                    const auto& bounds = content->InnerClipRect;
                    if (bounds.Contains(pointer)) {
                        const float edge      = 32 * scale;
                        const float direction = pointer.y < bounds.Min.y + edge ? -1.0F : pointer.y > bounds.Max.y - edge ? 1.0F : 0.0F;
                        if (direction) {
                            ImGui::SetScrollY(std::clamp(content->Scroll.y + direction * 450 * scale * std::min(ImGui::GetIO().DeltaTime, 0.05F), 0.0F, content->ScrollMax.y));
                            workspace.animate_until = workspace.frame_time + 0.1;
                        }
                    }
                }
            }
            ImGui::EndChild();
            ImGui::PopStyleVar();
        }
        ImGui::End();
        if (left) {
            ImGui::PopStyleColor(9);
            ImGui::PopStyleVar(5);
        }
        ImGui::PopStyleVar(2);
        if (selected) workspace.select_collection(std::move(*selected));
    }
} // namespace genesia::editor
