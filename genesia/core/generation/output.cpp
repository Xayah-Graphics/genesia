module;
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
module genesia.generation.output;
import genesia.io.files;

import std;

namespace genesia {
    namespace {
        constexpr auto crc_table = [] {
            std::array<std::uint32_t, 256> table{};
            for (std::uint32_t i = 0; i < table.size(); ++i) {
                auto crc = i;
                for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (crc & 1 ? 0xedb88320u : 0u);
                table[i] = crc;
            }
            return table;
        }();

        void write_chunk(std::string& image, const std::string_view type, const std::string_view data) {
            const auto length = std::byteswap(static_cast<std::uint32_t>(data.size()));
            std::uint32_t crc = 0xffffffffu;
            for (const auto bytes : {type, data})
                for (const unsigned char byte : bytes) crc = crc_table[(crc ^ byte) & 0xff] ^ (crc >> 8);
            crc = std::byteswap(~crc);
            image.append(reinterpret_cast<const char*>(&length), sizeof(length));
            image.append(type);
            image.append(data);
            image.append(reinterpret_cast<const char*>(&crc), sizeof(crc));
        }
    } // namespace

    std::filesystem::path save_image(const sdxl::Output& output, const Record& record, const std::filesystem::path& directory) {
        const auto model = record.model.u8string();
        nlohmann::json metadata{{"version", 2}, {"model", std::string{model.begin(), model.end()}}, {"seed", record.seed}, {"steps", record.parameters.steps}, {"cfg", record.parameters.cfg}, {"sampler", "euler"}, {"scheduler", "simple"}, {"prompt", {{"positive", record.parameters.positive}, {"negative", record.parameters.negative}}}};
        if (!record.parameters.loras.empty()) metadata["loras"] = record.parameters.loras;
        std::string text{"genesia"};
        // iTXt: keyword terminator, compression flag/method, empty language and translated keyword.
        text.append(5, '\0');
        text += metadata.dump();
        int length{};
        const std::unique_ptr<unsigned char, decltype(&std::free)> png{stbi_write_png_to_mem(output.pixels.data(), output.width * 3, output.width, output.height, 3, &length), &std::free};
        if (!png) throw std::runtime_error{"PNG encoding failed"};
        const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
        const auto path = directory / std::format("genesia_{:%Y%m%dT%H%M%SZ}_{}.png", now, files::random_id());
        std::string image;
        image.reserve(length + text.size() + 25);
        // stb writes the PNG signature and IHDR first; insert metadata before IDAT.
        image.append(reinterpret_cast<const char*>(png.get()), 33);
        write_chunk(image, "sRGB", std::string_view{"\0", 1});
        write_chunk(image, "iTXt", text);
        image.append(reinterpret_cast<const char*>(png.get() + 33), length - 33);
        files::write_bytes(path, {reinterpret_cast<const std::uint8_t*>(image.data()), image.size()}, false);
        return path;
    }

} // namespace genesia
