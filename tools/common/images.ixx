export module tools.images;
import std;
export namespace tools {
    struct ImageInfo final {
        int width{}, height{};
    };
    ImageInfo read_image_info(const std::filesystem::path& path);
    struct Image final {
        int width{}, height{};
        std::vector<std::uint8_t> pixels;
    };
    Image read_image(const std::filesystem::path& path);
} // namespace tools
