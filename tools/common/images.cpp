module;
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

module tools.images;
import std;

namespace tools {
    ImageInfo read_image_info(const std::filesystem::path& path) {
        std::ifstream file{path, std::ios::binary};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::array<unsigned char, 8> signature;
        file.read(reinterpret_cast<char*>(signature.data()), signature.size());
        if (signature != std::array<unsigned char, 8>{137, 80, 78, 71, 13, 10, 26, 10}) throw std::runtime_error{std::format("Not a PNG: {}", path.string())};
        ImageInfo image;
        file.seekg(16);
        std::array<std::uint32_t, 2> dimensions;
        file.read(reinterpret_cast<char*>(dimensions.data()), 8);
        image.width  = static_cast<int>(std::byteswap(dimensions[0]));
        image.height = static_cast<int>(std::byteswap(dimensions[1]));
        return image;
    }

    Image read_image(const std::filesystem::path& path) {
        std::ifstream file{path, std::ios::binary | std::ios::ate};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::vector<std::uint8_t> encoded(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(encoded.data()), encoded.size());
        Image result;
        const std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels{stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &result.width, &result.height, nullptr, 3), &stbi_image_free};
        if (!pixels) throw std::runtime_error{stbi_failure_reason()};
        result.pixels.assign(pixels.get(), pixels.get() + std::size_t(result.width) * result.height * 3);
        return result;
    }
    Image read_thumbnail(const std::filesystem::path& path, const int extent) {
        const auto source = read_image(path);
        const float scale = std::min(1.0F, float(extent) / std::max(source.width, source.height));
        Image result{.width = std::max(1, int(std::round(source.width * scale))), .height = std::max(1, int(std::round(source.height * scale)))};
        result.pixels.resize(std::size_t(result.width) * result.height * 3);
        if (!stbir_resize_uint8_srgb(source.pixels.data(), source.width, source.height, 0, result.pixels.data(), result.width, result.height, 0, STBIR_RGB)) throw std::runtime_error{"Cannot resize image: " + path.string()};
        return result;
    }
} // namespace tools
