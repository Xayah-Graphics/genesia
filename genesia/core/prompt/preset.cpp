module;
#include <nlohmann/json.hpp>

module genesia.prompt.preset;
import genesia.generation.defaults;
import std;

namespace genesia::prompt {
    Preset read_preset(const std::string_view name, const Catalog& catalog) {
        std::ifstream file{std::filesystem::path{defaults::assets} / "prompts" / (std::string{name} + ".json")};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        const auto json = nlohmann::json::parse(file);
        Preset result{std::string{name}, {}};
        for (const auto& [key, side] : {std::pair{"positive", &result.prompt.positive}, std::pair{"negative", &result.prompt.negative}}) {
            const auto& input = json.at(key);
            side->fixed       = input.at("fixed");
            for (const auto& source : input.at("groups")) {
                auto& group = side->groups.emplace_back(std::vector<Tag>{}, source.at("enabled").get<bool>());
                for (const auto& tag : source.at("tags")) {
                    const auto value = parse_tag(catalog, tag.get<std::string>());
                    if (!value) throw std::invalid_argument{std::format("{}: {}", key, value.error().message)};
                    const auto id = catalog.resolve(value->name);
                    if (!id) throw std::invalid_argument{std::format("{}: {}", key, id.error())};
                    group.tags.push_back({*id, value->weight});
                }
            }
        }
        return result;
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
        std::ofstream file{std::filesystem::path{defaults::assets} / "prompts" / (preset.name + ".json"), std::ios::out | (replace ? std::ios::trunc : std::ios::noreplace)};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << json.dump(2) << '\n';
        file.close();
    }

    std::vector<std::string> list_presets() {
        std::vector<std::string> result;
        for (const auto& entry : std::filesystem::directory_iterator{std::filesystem::path{defaults::assets} / "prompts"})
            if (entry.is_regular_file() && entry.path().extension() == ".json") result.push_back(entry.path().stem().string());
        std::ranges::sort(result);
        return result;
    }
} // namespace genesia::prompt
