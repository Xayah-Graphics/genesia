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
        if (!record.source.empty()) {
            const auto source   = record.source.u8string();
            metadata["repaint"] = {{"source", std::string{source.begin(), source.end()}}, {"denoise", record.parameters.denoise}};
        }
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

        std::filesystem::path path, temporary;
        std::ofstream file;
        for (;;) {
            path = directory / std::format("genesia_{:06}.png", next_index++);
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
            std::filesystem::create_hard_link(temporary, path);
            std::filesystem::remove(temporary);
        } catch (...) {
            file.exceptions(std::ios::goodbit);
            file.close();
            std::filesystem::remove(temporary);
            throw;
        }
        std::println("SAVE {} {:.3f}s", path.string(), std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        std::cout.flush();
        return path;
    }

    std::optional<Record> read_record(const std::filesystem::path& path, std::shared_ptr<const prompt::Catalog> catalog) {
        std::ifstream file{path, std::ios::binary};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.seekg(8);
        int width{}, height{};
        nlohmann::json metadata;
        for (;;) {
            std::uint32_t length{};
            std::array<char, 4> type;
            file.read(reinterpret_cast<char*>(&length), 4);
            file.read(type.data(), 4);
            length = std::byteswap(length);
            const std::string_view chunk{type.data(), type.size()};
            if (chunk == "IEND") break;
            if (chunk == "IHDR") {
                std::array<std::uint32_t, 2> dimensions;
                file.read(reinterpret_cast<char*>(dimensions.data()), 8);
                width  = static_cast<int>(std::byteswap(dimensions[0]));
                height = static_cast<int>(std::byteswap(dimensions[1]));
                file.seekg(length - 8, std::ios::cur);
            } else if (chunk == "iTXt") {
                std::string text(length, '\0');
                file.read(text.data(), text.size());
                if (text.starts_with(std::string_view{"genesia\0", 8})) {
                    if (!text.starts_with(std::string_view{"genesia\0\0\0\0\0", 12})) throw std::runtime_error{"Unsupported Genesia PNG metadata encoding"};
                    metadata = nlohmann::json::parse(text.begin() + 12, text.end());
                    break;
                }
            } else file.seekg(length, std::ios::cur);
            file.seekg(4, std::ios::cur);
        }
        if (metadata.is_null()) return {};
        if (metadata.at("version") != 1) throw std::runtime_error{"Unsupported Genesia PNG metadata version"};
        std::vector<prompt::ArchivedTag> archived;
        for (const auto& side : metadata.at("prompt"))
            for (const auto& group : side.at("groups"))
                for (const auto& tag : group.at("tags")) {
                    const std::string name = tag.at("name");
                    auto text              = tag.value("text", name);
                    if (!tag.contains("text")) std::ranges::replace(text, '_', ' ');
                    const auto id = catalog->resolve(name);
                    if ((!id || catalog->tags[*id].name != name || catalog->tags[*id].text != text) && !std::ranges::contains(archived, name, &prompt::ArchivedTag::name)) archived.push_back({name, std::move(text)});
                }
        if (!archived.empty()) catalog = std::make_shared<prompt::Catalog>(std::move(catalog), archived);
        Record result;
        result.path              = path;
        result.catalog           = std::move(catalog);
        result.parameters.width  = width;
        result.parameters.height = height;
        result.parameters.steps  = metadata.at("steps");
        result.parameters.cfg    = metadata.at("cfg");
        result.seed              = metadata.at("seed");
        const std::string model  = metadata.at("model");
        result.model             = std::filesystem::path{std::u8string{model.begin(), model.end()}};
        if (metadata.contains("repaint")) {
            const auto& repaint       = metadata.at("repaint");
            const std::string source  = repaint.at("source");
            result.source             = std::filesystem::path{std::u8string{source.begin(), source.end()}};
            result.parameters.denoise = repaint.at("denoise");
        }
        for (const auto& [name, side, text] : {std::tuple{"positive", &result.prompt.positive, &result.parameters.positive}, std::tuple{"negative", &result.prompt.negative, &result.parameters.negative}}) {
            const auto& saved = metadata.at("prompt").at(name);
            *text             = saved.at("text");
            side->fixed       = saved.at("fixed");
            for (const auto& source : saved.at("groups")) {
                auto& group = side->groups.emplace_back(std::vector<prompt::Tag>{}, source.at("enabled").get<bool>());
                for (const auto& tag : source.at("tags")) group.tags.push_back({result.catalog->resolve(tag.at("name").get<std::string>()).value(), tag.at("weight").get<float>()});
            }
        }
        return result;
    }
} // namespace genesia
