module;
#include <imgui.h>
export module genesia.editor.panels.application;
import genesia.editor.workspace;
import std;
export namespace genesia::editor {
    Workspace::ControlLayout control_layout(const Workspace& workspace, float scale, ImVec2 size, const Workspace::Picture& image);

    void generation_settings(Workspace& workspace, float scale, ImVec2 size);
    void top_strip(Workspace& workspace, float scale, ImVec2 size);
    void bottom_controls(Workspace& workspace, float scale, ImVec2 size, const Workspace::ControlLayout& layout, const Workspace::Picture& image);
} // namespace genesia::editor
