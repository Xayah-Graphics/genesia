export module genesia.editor.runtime.images;
import std;

export namespace genesia::editor {
    struct Image final {
        int width{};
        int height{};
        std::vector<std::uint8_t> pixels;
    };

    Image read_image(const std::filesystem::path& path);
    Image thumbnail(std::span<const std::uint8_t> pixels, int width, int height);
} // namespace genesia::editor
