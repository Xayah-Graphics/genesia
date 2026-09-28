export module edit.preset;
import std;
export namespace edit {
    struct Preset final {
        std::string name;
        std::string prompt;
        std::vector<std::filesystem::path> references;
    };
    Preset read_preset(std::string name);
    void write_preset(const Preset& preset, bool replace = true);
    std::vector<std::string> list_presets();
} // namespace edit
