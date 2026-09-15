module;
#include <nlohmann/json.hpp>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

module genesia.data.images;
import genesia.io.files;
import std;

namespace genesia {
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

    Record read_record(const std::filesystem::path& path) {
        const auto image = read_image_info(path);
        std::ifstream file{path, std::ios::binary};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.seekg(8);
        std::optional<nlohmann::json> stored;
        for (;;) {
            std::uint32_t length;
            std::array<char, 4> type;
            file.read(reinterpret_cast<char*>(&length), 4);
            file.read(type.data(), 4);
            length = std::byteswap(length);
            const std::string_view chunk{type.data(), type.size()};
            if (chunk == "IEND") break;
            if (chunk == "iTXt" || chunk == "tEXt" || chunk == "zTXt") {
                std::string text(length, '\0');
                file.read(text.data(), text.size());
                if (std::string_view{text}.substr(0, text.find('\0')) == "genesia") {
                    if (chunk != "iTXt" || !text.starts_with(std::string_view{"genesia\0\0\0\0\0", 12})) throw std::runtime_error{"Unsupported Genesia PNG metadata encoding"};
                    stored = nlohmann::json::parse(text.begin() + 12, text.end());
                }
            } else file.seekg(length, std::ios::cur);
            file.seekg(4, std::ios::cur);
        }
        if (!stored) throw std::runtime_error{"PNG has no Genesia generation record"};
        const auto& metadata     = *stored;
        Record result;
        result.path              = path;
        result.parameters.width  = image.width;
        result.parameters.height = image.height;
        if (metadata.at("version") != 2) throw std::runtime_error{"Unsupported Genesia PNG metadata version"};
        result.model            = files::path(metadata.at("model").get<std::string>());
        result.seed             = metadata.at("seed");
        result.parameters.steps = metadata.at("steps");
        result.parameters.cfg   = metadata.at("cfg");
        if (metadata.contains("loras")) metadata.at("loras").get_to(result.parameters.loras);
        result.parameters.positive = metadata.at("prompt").at("positive").get<std::string>();
        result.parameters.negative = metadata.at("prompt").at("negative").get<std::string>();
        if (metadata.contains("repaint")) {
            result.source             = files::path(metadata.at("repaint").at("source").get<std::string>());
            result.parameters.denoise = metadata.at("repaint").at("denoise");
        }
        return result;
    }


    Image read_image(const std::filesystem::path& path, const int channels) {
        std::ifstream file{path, std::ios::binary | std::ios::ate};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::vector<std::uint8_t> encoded(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(encoded.data()), encoded.size());
        Image result;
        int source_channels;
        const std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels{stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &result.width, &result.height, &source_channels, channels), &stbi_image_free};
        if (!pixels) throw std::runtime_error{stbi_failure_reason()};
        result.pixels.assign(pixels.get(), pixels.get() + std::size_t(result.width) * result.height * channels);
        return result;
    }
} // namespace genesia
