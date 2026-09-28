module;
#include <nlohmann/json.hpp>
module genesia.prompt.presets;
import genesia.io.files;
import std;

namespace genesia::prompt {
    namespace {
        const std::filesystem::path directory{std::filesystem::path{project::assets} / "prompts"};
    }

    Preset read_preset(std::string name, const Catalog& catalog) {
        return read_preset_file(directory / "presets" / files::path(name + ".json"), catalog);
    }
    Preset read_preset_file(const std::filesystem::path& source, const Catalog& catalog) {
        try {
            const auto json = files::read_json(source);
            Preset result{files::utf8(source.stem())};
            for (const auto& [key, side] : {std::pair{"positive", &result.prompt.positive}, std::pair{"negative", &result.prompt.negative}}) {
                const auto& input = json.at(key);
                side->fixed       = input.at("fixed").get<std::string>();
                for (const auto& item : input.at("groups")) {
                    auto& group = side->groups.emplace_back(std::vector<Tag>{}, item.at("enabled").get<bool>());
                    for (const auto& tag : item.at("tags")) {
                        const auto value = parse_tag(catalog, tag.get<std::string>());
                        if (!value) throw std::invalid_argument{value.error().message};
                        const auto id = catalog.resolve(value->name);
                        if (!id) throw std::invalid_argument{id.error()};
                        group.tags.push_back({*id, value->weight});
                    }
                }
            }
            return result;
        } catch (const std::exception& error) {
            throw std::runtime_error{std::format("{}: {}", files::utf8(source), error.what())};
        }
    }

    void write_preset(const Preset& preset, const Catalog& catalog, const bool replace) {
        nlohmann::json json;
        for (const auto& [key, side] : {std::pair{"positive", &preset.prompt.positive}, std::pair{"negative", &preset.prompt.negative}}) {
            auto& output     = json[key];
            output["fixed"]  = side->fixed;
            output["groups"] = nlohmann::json::array();
            for (const auto& group : side->groups) {
                nlohmann::json tags = nlohmann::json::array();
                for (const auto tag : group.tags) tags.push_back(serialize(catalog, std::span{&tag, 1}));
                output["groups"].push_back({{"enabled", group.enabled}, {"tags", std::move(tags)}});
            }
        }
        std::filesystem::create_directories(directory / "presets");
        std::ofstream file{directory / "presets" / files::path(preset.name + ".json"), std::ios::out | (replace ? std::ios::trunc : std::ios::noreplace)};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << json.dump(2) << '\n';
        file.close();
    }

    std::vector<std::string> list_presets() {
        std::vector<std::string> result;
        for (const auto& entry : std::filesystem::directory_iterator{directory / "presets"})
            if (entry.is_regular_file() && entry.path().extension() == ".json") result.push_back(files::utf8(entry.path().stem()));
        std::ranges::sort(result);
        return result;
    }

    void export_prompt(const std::string_view name, const std::array<std::string, 2>& text) {
        std::filesystem::create_directories(directory / "exports");
        std::ofstream file{directory / "exports" / files::path(std::string{name} + ".json")};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << nlohmann::json{{"positive", text[0]}, {"negative", text[1]}}.dump(2) << '\n';
        file.close();
    }
} // namespace genesia::prompt
