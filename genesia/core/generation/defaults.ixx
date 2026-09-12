export module genesia.generation.defaults;
import std;

export namespace genesia::defaults {
    inline constexpr int width               = 1024;
    inline constexpr int height              = 1536;
    inline constexpr int steps               = 50;
    inline constexpr float cfg               = 4.5F;
    inline constexpr float denoise           = 0.75F;
    inline constexpr bool random_seed        = true;
    inline constexpr std::uint64_t seed      = 16494404764960740964ULL;
    inline constexpr bool preview_enabled    = true;
    inline constexpr int preview_interval_ms = 1000;
#if defined(_WIN32)
    inline constexpr std::string_view checkpoint = "C:/Users/xayah/Documents/ComfyUI/models/checkpoints/oneObsession_v22.safetensors";
#elif defined(__linux__)
    inline constexpr std::string_view checkpoint = "/workspace/models/oneObsession_v22.safetensors";
#endif
    inline constexpr std::string_view output = "data/sdxl";
    inline constexpr std::string_view cache  = "genesia-cache";
    inline constexpr std::string_view preset = "default";
    inline constexpr std::string_view assets = GENESIA_ASSET_DIRECTORY;
} // namespace genesia::defaults
