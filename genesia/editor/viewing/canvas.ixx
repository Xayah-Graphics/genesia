module;
#include <imgui.h>
export module genesia.editor.viewing.canvas;
import genesia.editor.workspace;
import std;
export namespace genesia::editor {
    std::vector<Workspace::ImageResult> image_results(const Workspace& workspace, const Workspace::Picture& image);
    Workspace::ImageAction image_panel(Workspace& workspace, const char* id, const Workspace::Picture& image, ImVec2 origin, ImVec2 size, float scale, bool interactive, float brightness = 1);
    void canvas(Workspace& workspace, float scale, ImVec2 size);
} // namespace genesia::editor
