export module genesia.editor;
export import genesia.prompt.preset;
import std;
export namespace genesia::editor {
    void run(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog);
}
