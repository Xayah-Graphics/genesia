export module genesia.generation.output;

import genesia.sdxl;
import genesia.prompt;
import std;

export namespace genesia {
    struct Record final {
        sdxl::Parameters parameters;
        std::uint64_t seed{};
        std::filesystem::path directory;
        double load_seconds{};
        double prepare_seconds{};
        double sample_seconds{};
        double decode_seconds{};
        std::size_t resident_bytes{};
        std::size_t cache_hits{};
        std::size_t cache_misses{};
        std::chrono::steady_clock::time_point generation_started;
        prompt::Pair prompt;
        std::shared_ptr<const prompt::Catalog> catalog;
    };
    struct Image final {
        int width{};
        int height{};
        std::vector<std::uint8_t> pixels;
    };
    std::filesystem::path session_directory(const std::filesystem::path& root);
    void save(const sdxl::Output& output, const Record& record);
    Image read_image(const std::filesystem::path& path);
    Image thumbnail(const sdxl::Output& output);
}
