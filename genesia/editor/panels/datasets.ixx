module;
#include <imgui.h>
export module genesia.editor.panels.datasets;
import genesia.editor.workspace;
import genesia.runtime.tasks;
import std;
export namespace genesia::editor {
    void dataset_controls(Workspace& workspace, float scale);
    std::string concept_activity(const Workspace& workspace, std::string_view key);
    std::optional<std::string> dataset_contents(Workspace& workspace);
    void operation_activity(Workspace& workspace, std::initializer_list<runtime::Kind> kinds, std::string_view key);
    void training_controls(Workspace& workspace, const training::TrainingData& source, float scale);
    void classify_controls(Workspace& workspace, const training::TrainingData& source);
    void audit_controls(Workspace& workspace, const training::TrainingData& source);
} // namespace genesia::editor
