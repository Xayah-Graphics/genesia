export module genesia.generation.images;
import std;

export namespace genesia {
    struct Image final {
        int width{};
        int height{};
        std::vector<std::uint8_t> pixels;
    };

    Image read_image(const std::filesystem::path& path);
} // namespace genesia
