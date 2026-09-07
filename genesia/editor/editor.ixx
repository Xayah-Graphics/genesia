export module genesia.editor;
export import genesia.generation.configuration;
import std;
export namespace genesia::editor {
    void run(Configuration configuration, const std::filesystem::path& configuration_path);
}
