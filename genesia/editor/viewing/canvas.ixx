module;
#include <imgui.h>
export module genesia.editor.viewing.canvas;
import genesia.editor.workspace;
import std;
export namespace genesia::editor {
    bool image_panel(Workspace& workspace, const char* id, const Workspace::Picture& image, ImVec2 origin, ImVec2 size, bool interactive, float brightness = 1);
    void canvas(Workspace& workspace, float scale, ImVec2 size);
} // namespace genesia::editor
