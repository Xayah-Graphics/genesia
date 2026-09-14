export module genesia.headless;
export import genesia.generation.defaults;
export import genesia.prompt.preset;
export import genesia.work;
import std;
export namespace genesia::headless {
    struct Options final {
        work::Request request;
        int count{1};
        std::optional<std::uint64_t> first_seed;
        std::filesystem::path source;
        std::optional<int> width, height, steps;
        std::optional<float> cfg, denoise;
        std::vector<std::string> activated;
    };
    int run(const std::optional<prompt::Preset>& preset, std::shared_ptr<const prompt::Catalog> catalog, const Options& options);
} // namespace genesia::headless
