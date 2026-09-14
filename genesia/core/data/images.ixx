export module genesia.data.images;
export import genesia.generation.settings;
export import genesia.prompt.document;
import std;

export namespace genesia {
    struct Record final {
        generation::Settings parameters;
        std::uint64_t seed{};
        std::filesystem::path path, model;
        prompt::Document prompt;
        std::filesystem::path source;
    };
    Record read_record(const std::filesystem::path& path);
    struct Image final {
        int width{};
        int height{};
        std::vector<std::uint8_t> pixels;
    };

    Image read_image(const std::filesystem::path& path);
} // namespace genesia
