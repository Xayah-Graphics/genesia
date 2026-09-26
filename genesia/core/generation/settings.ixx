module;
#include <nlohmann/json.hpp>
export module genesia.generation.settings;
import std;

export namespace genesia::defaults {
    inline constexpr int width               = 1024;
    inline constexpr int height              = 1536;
    inline constexpr int steps               = 50;
    inline constexpr float cfg               = 4.5F;
    inline constexpr float lora_weight       = 0.95F;
    inline constexpr float lora_start        = 0.1F;
    inline constexpr bool random_seed        = true;
    inline constexpr std::uint64_t seed      = 16494404764960740964ULL;
    inline constexpr bool preview_enabled    = true;
    inline constexpr int preview_interval_ms = 1000;
    inline constexpr std::string_view preset = "default";
} // namespace genesia::defaults

export namespace genesia::generation {
    struct Lora final {
        std::string file, sha;
        float weight{defaults::lora_weight};
        float start{defaults::lora_start};
        bool operator==(const Lora&) const = default;
    };
    inline void to_json(nlohmann::json& json, const Lora& value) {
        json = {{"file", value.file}, {"sha", value.sha}, {"weight", value.weight}, {"start", value.start}};
    }
    struct Settings final {
        std::string positive;
        std::string negative;
        int width{defaults::width};
        int height{defaults::height};
        int steps{defaults::steps};
        float cfg{defaults::cfg};
        std::vector<Lora> loras;
        bool operator==(const Settings&) const = default;
    };

    std::string positive_prompt(const Settings& parameters);
} // namespace genesia::generation
