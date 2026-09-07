module;

#include <nlohmann/json.hpp>

module genesia.generation.configuration;
import std;

namespace genesia {
    Configuration read_configuration(const std::filesystem::path& path) {
        std::ifstream file{path};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        const auto json = nlohmann::json::parse(file);
        Configuration result{json.at("checkpoint").get<std::string>(), json.at("output").get<std::string>(), json.at("cache").get<std::string>(),
            {{}, {}, json.at("width"), json.at("height"), json.at("steps"), json.at("cfg")}, json.at("seeds").get<std::vector<std::uint64_t>>(), json.at("warmup"),
            {json.at("preview").at("enabled"), json.at("preview").at("interval_ms")}};
        for (const auto& tag : json.at("custom_tags")) result.custom_tags.push_back({tag.at("name"), tag.at("text")});
        result.catalog = std::make_shared<prompt::Catalog>(result.custom_tags);
        for (const auto& [name, side] : {std::pair{"positive", &result.prompt.positive}, std::pair{"negative", &result.prompt.negative}}) {
            const auto& input = json.at("prompt").at(name);
            side->fixed = input.at("fixed");
            for (const auto& source : input.at("groups")) {
                auto& group = side->groups.emplace_back(source.at("name").get<std::string>(), std::vector<prompt::Tag>{}, source.at("enabled").get<bool>());
                for (const auto& tag : source.at("tags")) {
                    const auto id = result.catalog->resolve(tag.at("name").get<std::string>());
                    if (!id) throw std::invalid_argument{std::format("prompt.{}: {}", name, id.error())};
                    group.tags.push_back({*id, tag.at("weight")});
                }
            }
        }
        result.parameters.positive = prompt::compose(*result.catalog, result.prompt.positive);
        result.parameters.negative = prompt::compose(*result.catalog, result.prompt.negative);
        return result;
    }

    void write_configuration(const Configuration& configuration, const std::filesystem::path& path) {
        const auto& p = configuration.parameters;
        nlohmann::json json{{"checkpoint", configuration.checkpoint.generic_string()}, {"output", configuration.output.generic_string()}, {"cache", configuration.cache.generic_string()},
            {"width", p.width}, {"height", p.height}, {"steps", p.steps}, {"cfg", p.cfg}, {"seeds", configuration.seeds}, {"warmup", configuration.warmup},
            {"preview", {{"enabled", configuration.preview.enabled}, {"interval_ms", configuration.preview.interval_ms}}}, {"custom_tags", nlohmann::json::array()}};
        for (const auto& tag : configuration.custom_tags) json["custom_tags"].push_back({{"name", tag.name}, {"text", tag.text}});
        for (const auto& [name, side] : {std::pair{"positive", &configuration.prompt.positive}, std::pair{"negative", &configuration.prompt.negative}}) {
            auto& output = json["prompt"][name];
            output["fixed"] = side->fixed;
            output["groups"] = nlohmann::json::array();
            for (const auto& group : side->groups) {
                nlohmann::json tags = nlohmann::json::array();
                for (const auto tag : group.tags) tags.push_back({{"name", configuration.catalog->tags[tag.id].name}, {"weight", tag.weight}});
                output["groups"].push_back({{"name", group.name}, {"enabled", group.enabled}, {"tags", std::move(tags)}});
            }
        }
        std::ofstream file{path};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << json.dump(2) << '\n';
    }
}
