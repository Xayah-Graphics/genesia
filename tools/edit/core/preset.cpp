module;
#include <nlohmann/json.hpp>
module edit.preset;
import tools.files;
import std;
namespace edit {
    namespace {
        const std::filesystem::path directory{std::filesystem::path{EDIT_ASSET_DIRECTORY} / "edit" / "preset"};
    }
    Preset read_preset(std::string name) {
        const auto file = directory / tools::files::path(name + ".json");
        try {
            const auto saved = tools::files::read_json(file);
            Preset preset{std::move(name), saved.at("prompt").get<std::string>()};
            for (const auto& path : saved.at("references")) preset.references.push_back(tools::files::path(path.get<std::string>()));
            return preset;
        } catch (const std::exception& failure) {
            throw std::runtime_error{tools::files::utf8(file) + ": " + failure.what()};
        }
    }
    void write_preset(const Preset& preset, const bool replace) {
        auto paths = nlohmann::json::array();
        for (const auto& path : preset.references) paths.push_back(tools::files::utf8(path));
        tools::files::write_json(directory / tools::files::path(preset.name + ".json"), {{"prompt", preset.prompt}, {"references", std::move(paths)}}, replace);
    }
    std::vector<std::string> list_presets() {
        std::vector<std::string> names;
        for (const auto& entry : std::filesystem::directory_iterator{directory})
            if (entry.is_regular_file() && entry.path().extension() == ".json") names.push_back(tools::files::utf8(entry.path().stem()));
        std::ranges::sort(names);
        return names;
    }
} // namespace edit
