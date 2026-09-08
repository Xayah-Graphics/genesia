module;
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

module genesia.editor.runtime.images;
import std;

namespace genesia::editor {
    Image read_image(const std::filesystem::path& path) {
        std::ifstream file{path, std::ios::binary | std::ios::ate};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::vector<std::uint8_t> encoded(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(encoded.data()), encoded.size());
        Image result;
        int channels;
        const std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels{stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &result.width, &result.height, &channels, 3), &stbi_image_free};
        if (!pixels) throw std::runtime_error{stbi_failure_reason()};
        result.pixels.assign(pixels.get(), pixels.get() + std::size_t(result.width) * result.height * 3);
        return result;
    }

    Image thumbnail(const std::span<const std::uint8_t> pixels, const int width, const int height) {
        Image result{192, std::max(1, height * 192 / width)};
        result.pixels.resize(std::size_t(result.width) * result.height * 3);
        stbir_resize_uint8_srgb(pixels.data(), width, height, 0, result.pixels.data(), result.width, result.height, 0, STBIR_RGB);
        return result;
    }
} // namespace genesia::editor
