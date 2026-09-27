export module genesia.project;
import std;
export namespace genesia::project {
    inline constexpr std::string_view assets = GENESIA_ASSET_DIRECTORY;
#if defined(_WIN32)
    inline const std::filesystem::path output{"C:/Users/xayah/Documents/ComfyUI/output"};
    inline constexpr std::string_view checkpoint = "C:/Users/xayah/Documents/ComfyUI/models/checkpoints/oneObsession_v22.safetensors";
#elif defined(__linux__)
    inline const std::filesystem::path output{"/workspace/output"};
    inline constexpr std::string_view checkpoint = "/workspace/models/oneObsession_v22.safetensors";
#endif
    inline constexpr std::string_view cache = GENESIA_CACHE_DIRECTORY;
} // namespace genesia::project
