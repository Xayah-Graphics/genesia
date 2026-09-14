module;
#include <imgui.h>
export module genesia.editor.widgets.controls;
import std;
export namespace genesia::editor {
    inline constexpr ImGuiWindowFlags overlay = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar;
    inline constexpr float top_strip_height = 48, control_height = 40, bottom_margin = 24;
    struct ParameterEdit final {
        ImGuiID id{};
        ImGuiDataType type{};
        void* value{};
    };
    void control_shade(ImVec2 minimum, ImVec2 maximum, float scale);
    void control_text(std::string_view text, ImVec2 position, float right, ImVec4 ink, float scale);
    bool text_button(const char* id, const char* label, float scale, float width = 0, bool selected = false, ImVec4 accent = {});
    void number_field(const char* id, const char* label, ImGuiDataType type, void* value, ImVec2 size, const void* step, const char* format, float scale, ParameterEdit& edit);
} // namespace genesia::editor
