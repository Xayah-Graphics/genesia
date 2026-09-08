export module genesia.generation.output;

import genesia.sdxl;
import genesia.prompt;
import std;

export namespace genesia {
    struct Record final {
        sdxl::Parameters parameters;
        std::uint64_t seed{};
        std::filesystem::path path;
        std::filesystem::path model;
        prompt::Pair prompt;
        std::shared_ptr<const prompt::Catalog> catalog;
    };
    struct Image final {
        int width{};
        int height{};
        std::vector<std::uint8_t> pixels;
    };
    struct ImageWriter final {
        std::filesystem::path directory;
        std::uint64_t next_index{1};

        explicit ImageWriter(std::filesystem::path directory);
        std::filesystem::path save(const sdxl::Output& output, const Record& record);
    };
    Image read_image(const std::filesystem::path& path);
    Image thumbnail(const sdxl::Output& output);
}
