module;
#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
module genesia.editor.widgets.captions;
import std;
namespace genesia::editor {
    bool CaptionEditor::draw(const prompt::TagSearch& search, const float scale) {
        const float width = ImGui::GetContentRegionAvail().x;
        const float inset = 8 * scale;
        const float lines = std::clamp(ImGui::CalcTextSize(input.c_str(), nullptr, false, width - 2 * inset).y / ImGui::GetTextLineHeight(), 3.0F, 5.0F);
        const ImVec2 size{width, lines * ImGui::GetTextLineHeight() + 2 * inset};
        const bool visible = ImGui::IsRectVisible(size);
        if (focus) {
            ImGui::SetKeyboardFocusHere();
            focus = false;
        }
        const auto input_id   = ImGui::GetID("##CaptionInput");
        const bool was_active = ImGui::GetActiveID() == input_id;
        const bool cancel     = was_active && ImGui::IsKeyPressed(ImGuiKey_Escape, false);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {inset, inset});
        const bool commit = ImGui::InputTextMultiline(
            "##CaptionInput", &input, size, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CtrlEnterForNewLine | ImGuiInputTextFlags_WordWrap | ImGuiInputTextFlags_CallbackAlways,
            [](ImGuiInputTextCallbackData* data) {
                static_cast<CaptionEditor*>(data->UserData)->cursor = data->CursorPos;
                return 0;
            },
            this);
        const bool active  = ImGui::IsItemActive();
        const auto minimum = ImGui::GetItemRectMin();
        const auto maximum = ImGui::GetItemRectMax();
        ImGui::PopStyleVar();
        if (active) editing = true;
        if (cancel) {
            input   = saved;
            editing = false;
            error.clear();
            suggestions.clear();
            return false;
        }
        const auto comma    = std::string_view{input}.substr(0, cursor).find_last_of(',');
        const auto begin    = comma == std::string::npos ? 0 : comma + 1;
        const auto end      = input.find(',', cursor);
        const auto fragment = std::string_view{input}.substr(begin, cursor - begin);
        if (query != fragment) {
            query       = fragment;
            suggestions = query.empty() ? std::vector<prompt::TagSuggestion>{} : search.search(query);
        }
        bool suggestions_hovered{};
        if (visible && editing && !suggestions.empty()) {
            const auto count       = std::min(std::size_t{5}, suggestions.size());
            const float row_height = ImGui::GetTextLineHeight() + 8 * scale;
            const float height     = row_height * count + 8 * scale;
            const auto* viewport   = ImGui::GetMainViewport();
            const float y          = maximum.y + height < viewport->WorkPos.y + viewport->WorkSize.y ? maximum.y : minimum.y - height;
            ImGui::SetNextWindowPos({minimum.x, std::max(viewport->WorkPos.y, y)});
            ImGui::SetNextWindowSize({size.x, height});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {inset, inset});
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 0});
            if (ImGui::Begin(std::format("##CaptionSuggestions{}", input_id).c_str(), nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNavFocus)) {
                ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
                for (std::size_t i = 0; i < count; ++i) {
                    ImGui::PushID(static_cast<int>(i));
                    const auto text = search.catalog.tags[suggestions[i].tag].text;
                    if (ImGui::Selectable("##Suggestion", false, ImGuiSelectableFlags_None, {0, row_height})) {
                        input.replace(begin, end == std::string::npos ? end : end - begin, (begin ? " " : "") + std::string{text});
                        focus = true;
                    }
                    const auto minimum = ImGui::GetItemRectMin();
                    const auto limit   = ImGui::GetItemRectMax();
                    ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), {minimum.x + inset, minimum.y + inset}, limit, limit.x, text.data(), text.data() + text.size(), nullptr);
                    ImGui::PopID();
                }
                suggestions_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
            }
            ImGui::End();
            ImGui::PopStyleVar(2);
        }
        if (commit || (editing && !active && !suggestions_hovered && !focus)) {
            editing = false;
            return input != saved;
        }
        return false;
    }
} // namespace genesia::editor
