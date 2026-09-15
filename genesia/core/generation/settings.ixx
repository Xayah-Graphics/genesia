module;
#include <nlohmann/json.hpp>
export module genesia.generation.settings;
export import genesia.generation.defaults;
import std;
export namespace genesia::generation {
    struct Lora final {
        std::string concept_key, sha;
        float weight{1};
        float start{};
        bool operator==(const Lora&) const = default;
    };
    inline void to_json(nlohmann::json& json, const Lora& value) {
        json = {{"concept", value.concept_key}, {"sha", value.sha}, {"weight", value.weight}, {"start", value.start}};
    }
    inline void from_json(const nlohmann::json& json, Lora& value) {
        json.at("concept").get_to(value.concept_key);
        json.at("sha").get_to(value.sha);
        json.at("weight").get_to(value.weight);
        value.start = json.value("start", 0.0F);
    }
    struct Settings final {
        std::string positive;
        std::string negative;
        int width{defaults::width};
        int height{defaults::height};
        int steps{defaults::steps};
        float cfg{defaults::cfg};
        float denoise{1};
        std::vector<Lora> loras;
        bool operator==(const Settings&) const = default;
    };

} // namespace genesia::generation
