module;
#include <imgui.h>
#include <imgui_internal.h>
module genesia.editor.panels.sidebars;
import genesia.editor.widgets.controls;
import genesia.editor.widgets.tags;
import genesia.prompt.library;
import std;
namespace genesia::editor {
    void preset_dialogs(Workspace& workspace, const float scale) {
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
                    prompts::write_preset(*workspace.prompt_library, next, *workspace.catalog, false);
                    workspace.preset_name = std::move(next.name);
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
            auto text = workspace.prompt_panel.composition.text;
            text[0]   = generation::positive_prompt(workspace.parameters());
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

    void sidebar(Workspace& workspace, const float scale, const ImVec2 size, const Workspace::Picture& image) {
        const auto& panel       = workspace.prompt_sidebar;
        const bool preview_drop = workspace.prompt_panel.incoming.has_value();
        const bool temporary    = preview_drop && workspace.page != Workspace::Page::generation;
        const bool open         = panel.open || preview_drop;
        if (!open || workspace.page != Workspace::Page::generation) workspace.prompt_editor.suspend();
        if (panel.amount == 0) return;
        const float visible = panel.width * panel.amount;
        const float top     = (top_strip_height + 24) * scale;
        ImGui::SetNextWindowPos({size.x - visible, top});
        ImGui::SetNextWindowSize({panel.width, std::max(1.0F, size.y - top - (bottom_margin + control_height + 16) * scale)});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16 * scale, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, panel.amount);
        if (ImGui::Begin("##PromptSidebar", nullptr, overlay | ImGuiWindowFlags_NoBackground | (open ? ImGuiWindowFlags_None : ImGuiWindowFlags_NoInputs))) {
            ImGui::PushFont(nullptr, 12);
            const float content_y = ImGui::GetCursorPosY() + ImGui::GetFontSize() + 12 * scale;
            if (temporary) ImGui::TextDisabled("SET PREVIEW");
            else if (workspace.page == Workspace::Page::generation) {
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 0});
                ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, {0, 0.5F});
                ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                if (ImGui::Button(workspace.preset_name.c_str(), {ImGui::CalcTextSize(workspace.preset_name.c_str()).x + 16 * scale, 0})) {
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
                if (ImGui::BeginPopup("Prompt presets")) {
                    for (const auto& name : workspace.preset_names)
                        if (ImGui::Selectable(name.c_str(), name == workspace.preset_name) && name != workspace.preset_name) workspace.switch_preset(name);
                    ImGui::Separator();
                    if (ImGui::MenuItem("Save as...")) workspace.save_as_requested = true;
                    ImGui::EndPopup();
                }
            } else ImGui::TextDisabled("Image prompt");
            ImGui::PopFont();
            ImGui::SetCursorPosY(content_y);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            workspace.prompt_panel.status();
            if (ImGui::BeginChild(temporary ? "##PreviewDropContent" : "##SidebarContent", {0, 0}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
                if (workspace.page == Workspace::Page::generation || temporary) {
                    workspace.prompt_panel.draw(workspace.prompt, *workspace.catalog, scale);
                    if (!temporary) workspace.prompt_editor.draw(workspace.prompt.free, workspace.tag_search, *workspace.catalog, scale);
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
        ImGui::PopStyleVar(2);
    }
} // namespace genesia::editor
