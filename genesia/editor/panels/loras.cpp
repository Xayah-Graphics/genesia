module;
#include <imgui.h>
module genesia.editor.panels.loras;
import genesia.editor.widgets.controls;
import genesia.io.files;
import std;
namespace genesia::editor {
    void lora_controls(Workspace& workspace, const float scale, const ImVec2 size, const Workspace::ControlLayout& layout) {
        if (workspace.page != Workspace::Page::generation) return;
        const float margin    = bottom_margin * scale;
        const float row       = control_height * scale;
        const float available = size.x - 3 * margin - layout.right_width;
        const bool stacked    = available < 480 * scale;
        const float width     = std::min(560 * scale, stacked ? size.x - 2 * margin : available);
        const float bottom    = size.y - margin - (stacked ? row + (layout.image_above ? row : 0) + 12 * scale : 0);
        const float height    = std::min(std::max(1.0F, float(workspace.loras.size())) * row, std::min(240 * scale, bottom - (top_strip_height + 24) * scale));
        ImGui::SetNextWindowPos({margin, bottom - height});
        ImGui::SetNextWindowSize({width, height});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8 * scale, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 5 * scale);
        ImGui::Begin("##Loras", nullptr, (overlay & ~ImGuiWindowFlags_NoScrollbar) | ImGuiWindowFlags_NoBackground);
        ImGui::PopStyleVar(3);
        if (workspace.loras.empty()) {
            ImGui::TextDisabled("LoRA · assets/loras");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Place .safetensors files directly in assets/loras, then reopen Genesia.");
        }
        for (auto& lora : workspace.loras) {
            ImGui::PushID(lora.file.c_str());
            const auto origin      = ImGui::GetCursorScreenPos();
            const auto name        = files::utf8(files::path(lora.file).stem());
            const float name_x     = origin.x + 20 * scale;
            const float name_right = std::min(name_x + ImGui::CalcTextSize(name.c_str()).x, origin.x + ImGui::GetContentRegionAvail().x - 270 * scale);
            const float strength_x = name_right + 16 * scale;
            const float start_x    = strength_x + 142 * scale;
            const float right      = start_x + 112 * scale;
            const bool hovered     = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect(origin, {right, origin.y + row});
            if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
                workspace.commit_parameters();
                lora.active           = !lora.active;
                workspace.loras_dirty = true;
            }
            const bool editing   = workspace.parameter_edit.id && (workspace.parameter_edit.value == &lora.weight || workspace.parameter_edit.value == &lora.start);
            const bool expanded  = hovered || editing;
            auto* storage        = ImGui::GetStateStorage();
            const auto reveal_id = ImGui::GetID("##Reveal");
            const float target   = float(expanded);
            float reveal         = std::lerp(storage->GetFloat(reveal_id), target, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.045F));
            if (std::abs(reveal - target) < 0.01F) reveal = target;
            else workspace.animate_until = std::max(workspace.animate_until, workspace.frame_time + 0.12);
            storage->SetFloat(reveal_id, reveal);

            const float compact_right = name_right + 8 * scale;
            if (ImGui::IsRectVisible(origin, {right, origin.y + row})) control_shade(origin, {std::lerp(compact_right, right, reveal), origin.y + row}, scale);
            ImGui::GetWindowDrawList()->AddCircleFilled({origin.x + 6 * scale, origin.y + row / 2}, 3 * scale, ImGui::GetColorU32(lora.active ? ImVec4{0.40F, 0.78F, 0.57F, 1} : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled)));
            ImGui::SetCursorScreenPos({name_x, origin.y});
            ImGui::Dummy({std::max(1.0F, name_right - name_x), row});
            control_text(name, {name_x, origin.y + (row - ImGui::GetFontSize()) / 2}, name_right, ImGui::GetStyleColorVec4(lora.active ? ImGuiCol_Text : ImGuiCol_TextDisabled), scale);
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) ImGui::SetTooltip("%s\nMiddle-click: %s\nTrigger: %s", lora.file.c_str(), lora.active ? "disable" : "enable", name.c_str());
            if (reveal > 0) {
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * reveal);
                ImGui::BeginDisabled(!expanded);
                constexpr float step = .05F, percent = 1;
                ImGui::SetCursorScreenPos({strength_x, origin.y});
                number_field("##Strength", "Strength", ImGuiDataType_Float, &lora.weight, {138 * scale, row}, &step, "%.2f", scale, workspace.parameter_edit);
                workspace.loras_dirty |= ImGui::IsItemEdited();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) ImGui::SetTooltip("LoRA strength for the next image");
                ImGui::SetCursorScreenPos({start_x, origin.y});
                number_field("##Start", "Start", ImGuiDataType_Float, &lora.start, {112 * scale, row}, &percent, "%.0f%%", scale, workspace.parameter_edit);
                workspace.loras_dirty |= ImGui::IsItemEdited();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) ImGui::SetTooltip("Start in the full denoising schedule\n0%%: all steps · 100%%: never");
                ImGui::EndDisabled();
                ImGui::PopStyleVar();
            }
            ImGui::SetCursorScreenPos({origin.x, origin.y + row});
            ImGui::PopID();
        }
        ImGui::End();
    }
} // namespace genesia::editor
