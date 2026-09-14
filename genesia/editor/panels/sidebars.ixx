module;
#include <imgui.h>
export module genesia.editor.panels.sidebars;
import genesia.editor.workspace;
import std;
export namespace genesia::editor {
    void preset_dialogs(Workspace& workspace, float scale);
    void sidebar(Workspace& workspace, bool left, float scale, ImVec2 size, const Workspace::Picture& image);
} // namespace genesia::editor
