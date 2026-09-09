module;
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

module genesia.editor.runtime.images;
import std;

namespace genesia::editor {
    Image thumbnail(const std::span<const std::uint8_t> pixels, const int width, const int height) {
        const float ratio = std::min(1.0F, 256.0F / std::max(width, height));
        Image result{std::max(1, static_cast<int>(width * ratio)), std::max(1, static_cast<int>(height * ratio))};
        result.pixels.resize(std::size_t(result.width) * result.height * 3);
        stbir_resize_uint8_srgb(pixels.data(), width, height, 0, result.pixels.data(), result.width, result.height, 0, STBIR_RGB);
        return result;
    }
} // namespace genesia::editor
