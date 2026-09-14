module;
#include <imgui.h>
#include <imgui_internal.h>
module genesia.editor.widgets.controls;
import std;
namespace genesia::editor {
    void control_shade(const ImVec2 minimum, const ImVec2 maximum, const float scale) {
        auto* draw          = ImGui::GetWindowDrawList();
        const float x       = (minimum.x + maximum.x) / 2;
        const float y       = (minimum.y + maximum.y) / 2;
        const float feather = 12 * scale;
        const ImVec2 first{minimum.x - feather, minimum.y - feather};
        const ImVec2 last{maximum.x + feather, maximum.y + feather};
        const ImU32 clear = IM_COL32(16, 17, 20, 0);
        const ImU32 shade = ImGui::GetColorU32(ImVec4{0.063F, 0.067F, 0.078F, 0.70F});
        draw->PushClipRect(first, last, false);
        draw->AddRectFilledMultiColor(first, {x, y}, clear, clear, shade, clear);
        draw->AddRectFilledMultiColor({x, first.y}, {last.x, y}, clear, clear, clear, shade);
        draw->AddRectFilledMultiColor({first.x, y}, {x, last.y}, clear, shade, clear, clear);
        draw->AddRectFilledMultiColor({x, y}, last, shade, clear, clear, clear);
        draw->PopClipRect();
    }

    void control_text(const std::string_view text, const ImVec2 position, const float right, const ImVec4 ink, const float scale) {
        auto* draw = ImGui::GetWindowDrawList();
        ImGui::PushStyleColor(ImGuiCol_Text, {0, 0, 0, 0.7F});
        ImGui::RenderTextEllipsis(draw, {position.x, position.y + scale}, {right, position.y + ImGui::GetFontSize() + scale}, right, text.data(), text.data() + text.size(), nullptr);
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, ink);
        ImGui::RenderTextEllipsis(draw, position, {right, position.y + ImGui::GetFontSize()}, right, text.data(), text.data() + text.size(), nullptr);
        ImGui::PopStyleColor();
    }

    bool text_button(const char* id, const char* label, const float scale, const float width, const bool selected, const ImVec4 accent) {
        const bool primary = accent.w > 0;
        const auto text    = ImGui::CalcTextSize(label);
        const auto origin  = ImGui::GetCursorScreenPos();
        const ImVec2 size{width ? width : text.x + 24 * scale, (primary ? top_strip_height : control_height) * scale};
        ImGui::PushStyleColor(ImGuiCol_NavCursor, {0, 0, 0, 0});
        const bool clicked = ImGui::InvisibleButton(id, size, ImGuiButtonFlags_EnableNav);
        ImGui::PopStyleColor();
        const bool disabled = ImGui::GetItemFlags() & ImGuiItemFlags_Disabled;
        const bool focused  = !disabled && ImGui::IsItemFocused() && ImGui::GetCurrentContext()->NavCursorVisible;
        const bool hovered  = ImGui::IsItemHovered() || focused;
        auto* storage       = ImGui::GetStateStorage();
        const auto key      = ImGui::GetItemID();
        const float alpha   = disabled ? 0 : std::lerp(storage->GetFloat(key), hovered || selected ? 1.0F : 0.0F, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.12F));
        storage->SetFloat(key, alpha);
        auto* draw = ImGui::GetWindowDrawList();
        if (primary && alpha > 0) {
            const auto clear = ImGui::GetColorU32(ImVec4{accent.x, accent.y, accent.z, 0});
            const auto tint  = ImGui::GetColorU32(ImVec4{accent.x, accent.y, accent.z, alpha * 0.16F});
            draw->AddRectFilledMultiColor(origin, {origin.x + size.x, origin.y + size.y}, clear, tint, tint, clear);
        }
        const float visible = std::min(text.x, std::max(0.0F, size.x - 24 * scale));
        const ImVec2 position{origin.x + (size.x - visible) / 2, origin.y + (size.y - text.y) / 2};
        const ImVec4 ink = disabled ? ImVec4{0.62F, 0.62F, 0.62F, 1} : primary ? ImLerp(accent, ImVec4{1, 1, 1, 1}, alpha * 0.35F) : ImVec4{0.57F + 0.31F * alpha, 0.58F + 0.30F * alpha, 0.64F + 0.29F * alpha, 1};
        control_text(label, position, position.x + visible, ink, scale);
        if (focused) draw->AddLine({position.x, position.y + text.y + 3 * scale}, {position.x + visible, position.y + text.y + 3 * scale}, ImGui::GetColorU32(ink), scale);
        return clicked;
    }

    void number_field(const char* id, const char* label, const ImGuiDataType type, void* value, const ImVec2 size, const void* step, const char* format, const float scale, ParameterEdit& edit) {
        const auto origin = ImGui::GetCursorScreenPos();
        const ImVec2 end{origin.x + size.x, origin.y + size.y};
        const auto key     = ImGui::GetID(id);
        const bool active  = ImGui::GetActiveID() == key;
        const bool hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect(origin, end);
        auto* storage      = ImGui::GetStateStorage();
        const float alpha  = std::lerp(storage->GetFloat(key), active || hovered ? 1.0F : 0.0F, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.12F));
        storage->SetFloat(key, alpha);
        auto* draw        = ImGui::GetWindowDrawList();
        const auto text   = ImGui::CalcTextSize(label);
        const float inset = *label ? size.x - 56 * scale : 0;
        if (*label) draw->AddText({origin.x + 4 * scale, origin.y + (size.y - text.y) / 2}, ImGui::GetColorU32(ImVec4{0.57F + 0.22F * alpha, 0.58F + 0.22F * alpha, 0.64F + 0.22F * alpha, 1}), label);
        const bool clicked = hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
        if (clicked && ImGui::GetIO().MousePos.x < origin.x + inset) ImGui::SetKeyboardFocusHere();
        bool stepped{};
        if (active && step) {
            ImGui::SetKeyOwner(ImGuiKey_UpArrow, key);
            ImGui::SetKeyOwner(ImGuiKey_DownArrow, key);
            const bool up   = ImGui::IsKeyPressed(ImGuiKey_UpArrow, ImGuiInputFlags_Repeat, key);
            const bool down = ImGui::IsKeyPressed(ImGuiKey_DownArrow, ImGuiInputFlags_Repeat, key);
            if (up || down) {
                auto* input = ImGui::GetInputTextState(key);
                ImGui::DataTypeApplyFromText(input->TextA.Data, type, value, format);
                ImGui::DataTypeApplyOp(type, up ? '+' : '-', value, value, step);
                input->ReloadUserBufAndSelectAll();
                stepped = true;
            }
        }
        ImGui::SetCursorScreenPos({origin.x + inset, origin.y});
        ImGui::SetNextItemWidth(size.x - inset);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, {0, 0, 0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, {0.65F + 0.25F * alpha, 0.65F + 0.25F * alpha, 0.71F + 0.23F * alpha, 1});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {4 * scale, (size.y - ImGui::GetFontSize()) / 2});
        ImGui::InputScalar(id, type, value, nullptr, nullptr, format);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);
        if (ImGui::IsItemActive()) edit = {key, type, value};
        else if (edit.id == key) edit = {};
        if (ImGui::IsItemActive()) draw->AddLine({origin.x + inset + 4 * scale, end.y - 6 * scale}, {end.x - 4 * scale, end.y - 6 * scale}, ImGui::GetColorU32(ImVec4{0.65F, 0.60F, 0.88F, 0.8F}), scale);
        if (stepped) ImGui::MarkItemEdited(key);
    }

} // namespace genesia::editor
