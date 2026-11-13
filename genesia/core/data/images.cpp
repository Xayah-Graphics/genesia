module;
#if defined(_WIN32)
#include <Windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#endif
#include <nlohmann/json.hpp>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

module genesia.data.images;
import genesia.io.files;
import std;

namespace genesia {
    std::vector<std::filesystem::path> list_output_images() {
        std::filesystem::create_directories(project::output);
#if defined(_WIN32)
        std::vector<std::pair<std::uint64_t, std::filesystem::path>> images;
#else
        std::vector<std::pair<std::pair<std::int64_t, std::uint32_t>, std::filesystem::path>> images;
#endif
        for (const auto& entry : std::filesystem::directory_iterator{project::output}) {
            const auto& path = entry.path();
            auto extension   = files::utf8(path.extension());
            for (auto& letter : extension) if (letter >= 'A' && letter <= 'Z') letter = static_cast<char>(letter + 'a' - 'A');
            if (extension != ".png" || !entry.is_regular_file()) continue;
#if defined(_WIN32)
            WIN32_FILE_ATTRIBUTE_DATA attributes;
            if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &attributes)) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Read image creation time: " + files::utf8(path)};
            const auto created = (std::uint64_t{attributes.ftCreationTime.dwHighDateTime} << 32) | attributes.ftCreationTime.dwLowDateTime;
#else
            struct statx attributes;
            if (statx(AT_FDCWD, path.c_str(), 0, STATX_BTIME, &attributes)) throw std::system_error{errno, std::system_category(), "Read image creation time: " + files::utf8(path)};
            if (!(attributes.stx_mask & STATX_BTIME)) throw std::runtime_error{"Image creation time unavailable: " + files::utf8(path)};
            const std::pair created{attributes.stx_btime.tv_sec, attributes.stx_btime.tv_nsec};
#endif
            images.emplace_back(created, path);
        }
        std::ranges::sort(images);
        std::vector<std::filesystem::path> paths;
        paths.reserve(images.size());
        for (auto& [created, path] : images) paths.push_back(std::move(path));
        return paths;
    }

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
        const auto& metadata = *stored;
        Record result;
        result.path              = path;
        result.parameters.width  = image.width;
        result.parameters.height = image.height;
        if (metadata.at("version") != 2) throw std::runtime_error{"Unsupported Genesia PNG metadata version"};
        result.model               = files::path(metadata.at("model").get<std::string>());
        result.seed                = metadata.at("seed");
        result.parameters.steps    = metadata.at("steps");
        result.parameters.cfg      = metadata.at("cfg");
        result.parameters.positive = metadata.at("prompt").at("positive").get<std::string>();
        result.parameters.negative = metadata.at("prompt").at("negative").get<std::string>();
        return result;
    }
} // namespace genesia
