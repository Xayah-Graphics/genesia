module;
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
module genesia.generation.output;
import genesia.io.files;
import genesia.project;

import std;
import genesia.data.datasets;

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

    std::filesystem::path save_image(const sdxl::Output& output, const Record& record) {
        const auto started = std::chrono::steady_clock::now();
        const auto model   = record.model.u8string();
        nlohmann::json metadata{{"version", 1}, {"model", std::string{model.begin(), model.end()}}, {"seed", record.seed}, {"steps", record.parameters.steps}, {"cfg", record.parameters.cfg}, {"sampler", "euler"}, {"scheduler", "simple"}};
        if (!record.source.empty()) {
            const auto source   = record.source.u8string();
            metadata["repaint"] = {{"source", std::string{source.begin(), source.end()}}, {"denoise", record.parameters.denoise}};
        }
        for (const auto& [name, side, text] : {std::tuple{"positive", &record.prompt.sides[0], &record.parameters.positive}, std::tuple{"negative", &record.prompt.sides[1], &record.parameters.negative}}) {
            auto& saved     = metadata["prompt"][name];
            saved["text"]   = *text;
            saved["fixed"]  = side->fixed;
            saved["groups"] = nlohmann::json::array();
            for (const auto& group : side->groups) {
                nlohmann::json tags = nlohmann::json::array();
                for (const auto& tag : group.tags) {
                    const auto& entry = tag;
                    auto& saved_tag   = tags.emplace_back(nlohmann::json{{"name", entry.name}, {"weight", tag.weight}});
                    saved_tag["text"] = entry.text;
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

        const files::Lock files_lock{"image-moves"};
        const files::Lock publication{"raw-publish"};
        dataset::Index index;
        index.scan("raw");
        const auto& raw = index.roots.front();
        if (!raw.error.empty()) throw std::runtime_error{raw.error};
        if (!raw.conflicts.empty()) {
            std::string message{"Raw contains independent copies of the same image:"};
            for (const auto& conflict : raw.conflicts)
                for (const auto& path : conflict) message += "\n" + path.string();
            throw std::runtime_error{message};
        }
        std::uint64_t next_index{1};
        for (const auto& entry : std::filesystem::directory_iterator{project::raw}) {
            const auto filename = entry.path().filename().string();
            if (!filename.starts_with("genesia_") || !filename.ends_with(".png")) continue;
            const std::string_view digits{filename.data() + 8, filename.size() - 12};
            std::uint64_t number{};
            const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), number);
            if (parsed.ec == std::errc{} && parsed.ptr == digits.data() + digits.size()) next_index = std::max(next_index, number + 1);
        }
        std::filesystem::path path, temporary;
        std::ofstream file;
        for (;;) {
            path = project::raw / std::format("genesia_{:06}.png", next_index++);
            if (std::filesystem::exists(path)) continue;
            temporary = path;
            temporary += ".part";
            file.open(temporary, std::ios::binary | std::ios::noreplace);
            if (file.is_open()) break;
            if (!std::filesystem::exists(temporary)) throw std::runtime_error{std::format("Cannot create output image: {}", path.string())};
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
            // Publish a complete PNG atomically without replacing another process's output.
            const auto resource = index.identify(temporary);
            const auto existing = std::ranges::find(raw.all.images, resource.sha, &dataset::File::sha);
            std::filesystem::create_hard_link(existing == raw.all.images.end() ? temporary : existing->path, path);
            std::filesystem::remove(temporary);
        } catch (...) {
            file.exceptions(std::ios::goodbit);
            file.close();
            std::filesystem::remove(temporary);
            throw;
        }
        std::println(std::cerr, "SAVE {} {:.3f}s", path.string(), std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        return path;
    }

} // namespace genesia
