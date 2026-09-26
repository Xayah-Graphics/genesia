module;
#include <imgui.h>
export module genesia.editor.panels.loras;
import genesia.editor.workspace;
export namespace genesia::editor {
    void lora_controls(Workspace& workspace, float scale, ImVec2 size, const Workspace::ControlLayout& layout);
} // namespace genesia::editor
