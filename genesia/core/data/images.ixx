export module genesia.data.images;
export import genesia.generation.settings;
import std;

export namespace genesia {
    struct Record final {
        generation::Settings parameters;
        std::uint64_t seed{};
        std::filesystem::path path, model;
    };
    struct ImageInfo final {
        int width{}, height{};
    };
    ImageInfo read_image_info(const std::filesystem::path& path);
    struct Image final {
        int width{}, height{};
        std::vector<std::uint8_t> pixels;
    };
    Image read_image(const std::filesystem::path& path);
    Record read_record(const std::filesystem::path& path);
} // namespace genesia
