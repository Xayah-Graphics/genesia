module;
#include <nlohmann/json.hpp>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

module genesia.data.images;
import genesia.io.files;
import std;

namespace genesia {
    Record read_record(const std::filesystem::path& path) {
        std::ifstream file{path, std::ios::binary};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::array<unsigned char, 8> signature;
        file.read(reinterpret_cast<char*>(signature.data()), signature.size());
        if (signature != std::array<unsigned char, 8>{137, 80, 78, 71, 13, 10, 26, 10}) throw std::runtime_error{std::format("Not a PNG: {}", path.string())};
        Record result;
        result.path = path;
        nlohmann::json metadata;
        for (;;) {
            std::uint32_t length;
            std::array<char, 4> type;
            file.read(reinterpret_cast<char*>(&length), 4);
            file.read(type.data(), 4);
            length = std::byteswap(length);
            const std::string_view chunk{type.data(), type.size()};
            if (chunk == "IEND") break;
            if (chunk == "IHDR") {
                std::array<std::uint32_t, 2> dimensions;
                file.read(reinterpret_cast<char*>(dimensions.data()), 8);
                result.parameters.width  = static_cast<int>(std::byteswap(dimensions[0]));
                result.parameters.height = static_cast<int>(std::byteswap(dimensions[1]));
                file.seekg(length - 8, std::ios::cur);
            } else if (chunk == "iTXt") {
                std::string text(length, '\0');
                file.read(text.data(), text.size());
                if (text.starts_with(std::string_view{"genesia\0", 8})) {
                    if (!text.starts_with(std::string_view{"genesia\0\0\0\0\0", 12})) throw std::runtime_error{"Unsupported Genesia PNG metadata encoding"};
                    metadata = nlohmann::json::parse(text.begin() + 12, text.end());
                }
            } else file.seekg(length, std::ios::cur);
            file.seekg(4, std::ios::cur);
        }
        if (metadata.is_null()) throw std::runtime_error{std::format("Missing Genesia PNG metadata: {}", path.string())};
        if (metadata.at("version") != 1) throw std::runtime_error{"Unsupported Genesia PNG metadata version"};
        result.model            = files::path(metadata.at("model").get<std::string>());
        result.seed             = metadata.at("seed");
        result.parameters.steps = metadata.at("steps");
        result.parameters.cfg   = metadata.at("cfg");
        for (const auto& [name, side] : {std::pair{"positive", &result.prompt.sides[0]}, std::pair{"negative", &result.prompt.sides[1]}}) {
            const auto& saved = metadata.at("prompt").at(name);
            side->text        = saved.at("text");
            side->fixed       = saved.at("fixed");
            for (const auto& group : saved.at("groups").get_ref<const nlohmann::json::array_t&>()) {
                auto& parsed = side->groups.emplace_back(group.at("enabled").get<bool>());
                for (const auto& tag : group.at("tags").get_ref<const nlohmann::json::array_t&>()) {
                    auto& item = parsed.tags.emplace_back(tag.at("name").get<std::string>(), std::string{}, tag.at("weight").get<float>());
                    item.text  = tag.at("text");
                }
            }
        }
        if (metadata.contains("repaint")) {
            result.source             = files::path(metadata.at("repaint").at("source").get<std::string>());
            result.parameters.denoise = metadata.at("repaint").at("denoise");
        }
        result.parameters.positive = result.prompt.sides[0].text;
        result.parameters.negative = result.prompt.sides[1].text;
        return result;
    }


    Image read_image(const std::filesystem::path& path) {
        std::ifstream file{path, std::ios::binary | std::ios::ate};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::vector<std::uint8_t> encoded(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(encoded.data()), encoded.size());
        Image result;
        int source_channels;
        const std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels{stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &result.width, &result.height, &source_channels, 3), &stbi_image_free};
        if (!pixels) throw std::runtime_error{stbi_failure_reason()};
        result.pixels.assign(pixels.get(), pixels.get() + std::size_t(result.width) * result.height * 3);
        return result;
    }
} // namespace genesia
