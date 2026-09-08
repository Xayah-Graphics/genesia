export module genesia.prompt.preset;
export import genesia.prompt;
import std;

export namespace genesia::prompt {
    struct Preset final {
        std::string name;
        Pair prompt;
    };
    Preset read_preset(std::string_view name, const Catalog& catalog);
    void write_preset(const Preset& preset, const Catalog& catalog, bool replace = true);
    std::vector<std::string> list_presets();
} // namespace genesia::prompt
