export module genesia.prompt.presets;
export import genesia.prompt;
import std;

export namespace genesia::prompt {
    struct Preset final {
        std::string name;
        Pair prompt;
    };

    Preset read_preset(std::string name, const Catalog& catalog);
    Preset read_preset_file(const std::filesystem::path& source, const Catalog& catalog);
    void write_preset(const Preset& preset, const Catalog& catalog, bool replace = true);
    std::vector<std::string> list_presets();
    void export_prompt(std::string_view name, const std::array<std::string, 2>& text);
} // namespace genesia::prompt
