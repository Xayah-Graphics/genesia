module;
#include <genesia/cuda.h>

#include <nlohmann/json.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
module genesia.generation.output;

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

    ImageWriter::ImageWriter(std::filesystem::path root) : directory{std::move(root)} {
        std::filesystem::create_directories(directory);
        for (const auto& entry : std::filesystem::directory_iterator{directory}) {
            const auto filename = entry.path().filename().u8string();
            if (!filename.starts_with(u8"genesia_") || !filename.ends_with(u8".png")) continue;
            const std::string_view digits{reinterpret_cast<const char*>(filename.data()) + 8, filename.size() - 12};
            std::uint64_t index{};
            const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), index);
            if (parsed.ec == std::errc{} && parsed.ptr == digits.data() + digits.size()) next_index = std::max(next_index, index + 1);
        }
    }

    std::filesystem::path ImageWriter::save(const sdxl::Output& output, const Record& record) {
        const auto started = std::chrono::steady_clock::now();
        const auto model   = record.model.u8string();
        nlohmann::json metadata{{"version", 1}, {"model", std::string{model.begin(), model.end()}}, {"seed", record.seed}, {"steps", record.parameters.steps}, {"cfg", record.parameters.cfg}, {"sampler", "euler"}, {"scheduler", "simple"}};
        for (const auto& [name, side, text] : {std::tuple{"positive", &record.prompt.positive, &record.parameters.positive}, std::tuple{"negative", &record.prompt.negative, &record.parameters.negative}}) {
            auto& saved     = metadata["prompt"][name];
            saved["text"]   = *text;
            saved["fixed"]  = side->fixed;
            saved["groups"] = nlohmann::json::array();
            for (const auto& group : side->groups) {
                nlohmann::json tags = nlohmann::json::array();
                for (const auto tag : group.tags) {
                    const auto& entry = record.catalog->tags[tag.id];
                    auto& saved_tag   = tags.emplace_back(nlohmann::json{{"name", entry.name}, {"weight", tag.weight}});
                    if (entry.category == -1) saved_tag["text"] = entry.text;
                }
                saved["groups"].push_back({{"enabled", group.enabled}, {"tags", std::move(tags)}});
            }
        }
        std::string text{"genesia"};
        // iTXt: keyword terminator, compression flag/method, empty language and translated keyword.
        text.append(5, '\0');
        text += metadata.dump();
        int length{};
        const std::unique_ptr<unsigned char, decltype(&std::free)> png{stbi_write_png_to_mem(output.pixels.data(), output.width * 3, output.width, output.height, 3, &length), &std::free};
        if (!png) throw std::runtime_error{"PNG encoding failed"};

        std::filesystem::path path;
        std::ofstream file;
        for (;;) {
            path = directory / std::format("genesia_{:06}.png", next_index++);
            file.open(path, std::ios::binary | std::ios::noreplace);
            if (file.is_open()) break;
            if (!std::filesystem::exists(path)) throw std::runtime_error{std::format("Cannot create output image: {}", path.string())};
            file.clear();
        }
        file.exceptions(std::ios::badbit | std::ios::failbit);
        try {
            // stb writes the PNG signature and IHDR first; insert metadata before IDAT.
            file.write(reinterpret_cast<const char*>(png.get()), 33);
            write_chunk(file, "sRGB", std::string_view{"\0", 1});
            write_chunk(file, "iTXt", text);
            file.write(reinterpret_cast<const char*>(png.get() + 33), length - 33);
            file.close();
        } catch (...) {
            file.exceptions(std::ios::goodbit);
            file.close();
            std::filesystem::remove(path);
            throw;
        }
        std::println("SAVE {} {:.3f}s", path.string(), std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        std::cout.flush();
        return path;
    }
} // namespace genesia
