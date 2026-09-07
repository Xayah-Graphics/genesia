module;

#include <nlohmann/json.hpp>

module genesia.generation.configuration;

import std;

namespace genesia {
    Configuration read_configuration(const std::filesystem::path& path) {
        std::ifstream file{path};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        const auto json = nlohmann::json::parse(file);
        return {json.at("checkpoint").get<std::string>(), json.at("output").get<std::string>(), json.at("cache").get<std::string>(),
            {json.at("positive"), json.at("negative"), json.at("width"), json.at("height"), json.at("steps"), json.at("cfg")},
            json.at("seeds").get<std::vector<std::uint64_t>>(), json.at("warmup")};
    }
    void write_configuration(const Configuration& configuration, const std::filesystem::path& path) {
        const auto& p = configuration.parameters;
        const nlohmann::json json{{"checkpoint", configuration.checkpoint.generic_string()}, {"output", configuration.output.generic_string()}, {"cache", configuration.cache.generic_string()},
            {"positive", p.positive}, {"negative", p.negative}, {"width", p.width}, {"height", p.height}, {"steps", p.steps}, {"cfg", p.cfg}, {"seeds", configuration.seeds}, {"warmup", configuration.warmup}};
        std::ofstream file{path};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << json.dump(2) << '\n';
    }
}
