module;
#include <imgui.h>
#include <imgui_internal.h>
module tools.editor.style;

namespace tools::editor {
    void apply_style() {
        auto& context                     = *ImGui::GetCurrentContext();
        context.ConfigNavEnableTabbing    = false;
        context.ConfigNavWindowingKeyNext = 0;
        context.ConfigNavWindowingKeyPrev = 0;
        ImGui::StyleColorsDark();
        auto& style                           = ImGui::GetStyle();
        style.WindowPadding                   = {20, 20};
        style.FramePadding                    = {10, 7};
        style.ItemSpacing                     = {10, 10};
        style.ItemInnerSpacing                = {8, 6};
        style.WindowBorderSize                = 0;
        style.PopupBorderSize                 = 1;
        style.FrameBorderSize                 = 0;
        style.WindowRounding                  = 12;
        style.FrameRounding                   = 7;
        style.PopupRounding                   = 10;
        style.GrabRounding                    = 4;
        style.ScrollbarSize                   = 8;
        style.Colors[ImGuiCol_WindowBg]       = {0.095F, 0.106F, 0.128F, 1};
        style.Colors[ImGuiCol_PopupBg]        = {0.115F, 0.127F, 0.150F, 1};
        style.Colors[ImGuiCol_Text]           = {0.90F, 0.92F, 0.95F, 1};
        style.Colors[ImGuiCol_TextDisabled]   = {0.57F, 0.61F, 0.68F, 1};
        style.Colors[ImGuiCol_Border]         = {0.20F, 0.23F, 0.28F, 1};
        style.Colors[ImGuiCol_FrameBg]        = {0.14F, 0.16F, 0.19F, 1};
        style.Colors[ImGuiCol_FrameBgHovered] = {0.20F, 0.23F, 0.27F, 1};
        style.Colors[ImGuiCol_FrameBgActive]  = {0.25F, 0.28F, 0.32F, 1};
        style.Colors[ImGuiCol_Button]         = {0.18F, 0.21F, 0.25F, 1};
        style.Colors[ImGuiCol_ButtonHovered]  = {0.25F, 0.29F, 0.33F, 1};
        style.Colors[ImGuiCol_ButtonActive]   = {0.30F, 0.35F, 0.39F, 1};
        style.Colors[ImGuiCol_Header]         = style.Colors[ImGuiCol_Button];
        style.Colors[ImGuiCol_HeaderHovered]  = style.Colors[ImGuiCol_ButtonHovered];
        style.Colors[ImGuiCol_HeaderActive]   = style.Colors[ImGuiCol_ButtonActive];
        style.Colors[ImGuiCol_PlotLines]      = {0.61F, 0.75F, 0.72F, 1};
        style.Colors[ImGuiCol_PlotHistogram]  = style.Colors[ImGuiCol_PlotLines];
        style.Colors[ImGuiCol_CheckMark]      = style.Colors[ImGuiCol_PlotLines];
        style.Colors[ImGuiCol_SliderGrab]     = style.Colors[ImGuiCol_PlotLines];
        style.Colors[ImGuiCol_TextSelectedBg] = {0.35F, 0.50F, 0.48F, 0.65F};
    }

    void draw_welcome(const char* title, const char* subtitle) {
        const auto origin = ImGui::GetCursorScreenPos();
        const float width = ImGui::GetContentRegionAvail().x;
        const float scale = ImGui::GetStyle().FontScaleDpi;
        auto* draw        = ImGui::GetWindowDrawList();
        ImGui::PushFont(nullptr, 30);
        const auto text = ImGui::CalcTextSize(title);
        draw->AddText({origin.x + (width - text.x) / 2, origin.y + 28 * scale}, ImGui::GetColorU32(ImGuiCol_Text), title);
        ImGui::PopFont();
        const auto hint = ImGui::CalcTextSize(subtitle);
        draw->AddText({origin.x + (width - hint.x) / 2, origin.y + 78 * scale}, ImGui::GetColorU32(ImGuiCol_TextDisabled), subtitle);
        ImGui::Dummy({width, 120 * scale});
    }
} // namespace tools::editor
