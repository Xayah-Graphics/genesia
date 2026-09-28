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

        void write_chunk(std::ofstream& file, const std::string_view type, const std::string_view data) {
            const auto length = std::byteswap(static_cast<std::uint32_t>(data.size()));
            std::uint32_t crc = 0xffffffffu;
            for (const auto bytes : {type, data})
                for (const unsigned char byte : bytes) crc = crc_table[(crc ^ byte) & 0xff] ^ (crc >> 8);
            crc = std::byteswap(~crc);
            file.write(reinterpret_cast<const char*>(&length), sizeof(length));
            file.write(type.data(), type.size());
            file.write(data.data(), data.size());
            file.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
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
        static std::map<std::filesystem::path, std::uint64_t> numbers;
        const auto [next, first] = numbers.try_emplace(directory, 1);
        if (first) {
            std::filesystem::create_directories(directory);
            for (const auto& entry : std::filesystem::directory_iterator{directory}) {
                const auto name = files::utf8(entry.path().filename());
                if (!name.starts_with("genesia_") || !name.ends_with(".png")) continue;
                std::uint64_t value{};
                const auto parsed = std::from_chars(name.data() + 8, name.data() + name.size() - 4, value);
                if (parsed.ec == std::errc{} && parsed.ptr == name.data() + name.size() - 4) next->second = std::max(next->second, value + 1);
            }
        }
        const auto path = directory / std::format("genesia_{:06}.png", next->second++);
        auto temporary  = path;
        temporary += ".part";
        std::ofstream file{temporary, std::ios::binary | std::ios::trunc};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        try {
            // stb writes the PNG signature and IHDR first; insert metadata before IDAT.
            file.write(reinterpret_cast<const char*>(png.get()), 33);
            write_chunk(file, "sRGB", std::string_view{"\0", 1});
            write_chunk(file, "iTXt", text);
            file.write(reinterpret_cast<const char*>(png.get() + 33), length - 33);
            file.close();
            files::move(temporary, path);
        } catch (...) {
            file.exceptions(std::ios::goodbit);
            file.close();
            std::filesystem::remove(temporary);
            throw;
        }
        return path;
    }

} // namespace genesia
