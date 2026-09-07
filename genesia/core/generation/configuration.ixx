export module genesia.generation.configuration;

import genesia.sdxl;
export import genesia.prompt;
import std;

export namespace genesia {
    struct PreviewSettings final {
        bool enabled{true};
        int interval_ms{1000};
    };
    struct Configuration final {
        std::filesystem::path checkpoint;
        std::filesystem::path output;
        std::filesystem::path cache;
        sdxl::Parameters parameters;
        std::vector<std::uint64_t> seeds;
        int warmup{};
        PreviewSettings preview;
        std::vector<prompt::CustomTag> custom_tags;
        std::shared_ptr<const prompt::Catalog> catalog;
        prompt::Pair prompt;
    };
    Configuration read_configuration(const std::filesystem::path& path);
    void write_configuration(const Configuration& configuration, const std::filesystem::path& path);
} // namespace genesia
