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

    dataset::File save_image(dataset::Index& index, const sdxl::Output& output, const Record& record) {
        const auto model = record.model.u8string();
        nlohmann::json metadata{{"version", 1}, {"model", std::string{model.begin(), model.end()}}, {"seed", record.seed}, {"steps", record.parameters.steps}, {"cfg", record.parameters.cfg}, {"sampler", "euler"}, {"scheduler", "simple"}};
        if (!record.parameters.loras.empty()) metadata["loras"] = record.parameters.loras;
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
        const auto& raw = *std::ranges::find(index.roots, std::string_view{"raw"}, [](const dataset::Root& root) { return root.all.key; });
        if (!raw.ready) throw std::runtime_error{"Raw is not ready"};
        const auto path = project::raw / std::format("genesia_{:06}.png", index.next_output++);
        auto temporary  = path;
        temporary += ".part";
        std::ofstream file{temporary, std::ios::binary | std::ios::trunc};
        dataset::File resource;
        file.exceptions(std::ios::badbit | std::ios::failbit);
        try {
            // stb writes the PNG signature and IHDR first; insert metadata before IDAT.
            file.write(reinterpret_cast<const char*>(png.get()), 33);
            write_chunk(file, "sRGB", std::string_view{"\0", 1});
            write_chunk(file, "iTXt", text);
            file.write(reinterpret_cast<const char*>(png.get() + 33), length - 33);
            file.close();
            // Only complete files become dataset members.
            resource            = index.identify(temporary, std::array{output.width, output.height});
            const auto existing = std::ranges::find(raw.all.images, resource.sha, &dataset::File::sha);
            std::filesystem::create_hard_link(existing == raw.all.images.end() ? temporary : existing->path, path);
            if (existing != raw.all.images.end()) resource = *existing;
            resource.path = path;
            std::filesystem::remove(temporary);
        } catch (...) {
            file.exceptions(std::ios::goodbit);
            file.close();
            std::filesystem::remove(temporary);
            throw;
        }
        index.insert(resource);
        return resource;
    }

} // namespace genesia
