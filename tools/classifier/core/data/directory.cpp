module;
#if defined(_WIN32)
#include <Windows.h>
#endif
#include <nlohmann/json.hpp>
module classifier.data.directory;
import classifier.project;
import tools.hash;
import std;
namespace classifier::dataset {
    File read_file(const std::filesystem::path& path) {
        const auto info = tools::read_image_info(path);
        if (!((info.width == 1024 && info.height == 1536) || (info.width == 1536 && info.height == 1024) || (info.width == 1024 && info.height == 1024))) throw std::runtime_error{"Unsupported classifier image size: " + tools::files::utf8(path)};
        return {path, tools::files::digest(path), info.width, info.height};
    }
    Index::Index(const std::filesystem::path& root) {
        const auto name = tools::files::utf8(root);
        cache           = project::cache / "images" / (tools::sha256({reinterpret_cast<const unsigned char*>(name.data()), name.size()}) + ".json");
        if (!std::filesystem::exists(cache)) return;
        const auto stored = tools::files::read_json(cache);
        for (const auto& [name, value] : stored.items()) entries.emplace(name, Cached{value.at("modified"), value.at("bytes"), value.at("sha"), value.at("width"), value.at("height")});
    }
    File Index::identify(const std::filesystem::path& path) {
        const auto name = tools::files::utf8(path);
        visited.insert(name);
        const auto modified = std::filesystem::last_write_time(path).time_since_epoch().count();
        const auto bytes    = std::filesystem::file_size(path);
        auto found          = entries.find(name);
        if (found == entries.end() || found->second.modified != modified || found->second.bytes != bytes) {
            const auto file = read_file(path);
            found           = entries.insert_or_assign(name, Cached{modified, bytes, file.sha, file.width, file.height}).first;
            dirty           = true;
        }
        const auto& file = found->second;
        return {path, file.sha, file.width, file.height};
    }
    void Index::flush() {
        if (std::erase_if(entries, [&](const auto& entry) { return !visited.contains(entry.first); })) dirty = true;
        if (!dirty) return;
        auto stored = nlohmann::json::object();
        for (const auto& [name, file] : entries) stored[name] = {{"modified", file.modified}, {"bytes", file.bytes}, {"sha", file.sha}, {"width", file.width}, {"height", file.height}};
        tools::files::write_json(cache, stored);
        dirty = false;
    }
    MoveResult move_images(std::vector<Move> moves) {
        const auto ordered = [](const std::filesystem::path& a, const std::filesystem::path& b) {
#if defined(_WIN32)
            return CompareStringOrdinal(a.c_str(), -1, b.c_str(), -1, TRUE) == CSTR_LESS_THAN;
#else
            return a < b;
#endif
        };
        std::set<std::filesystem::path, decltype(ordered)> reserved, vacated;
        std::set<std::filesystem::path> directories;
        for (const auto& move : moves) {
            if (!reserved.insert(move.destination).second || (std::filesystem::exists(move.destination) && !vacated.contains(move.destination))) throw std::runtime_error{"Destination already exists: " + tools::files::utf8(move.destination)};
            vacated.insert(move.source);
            vacated.erase(move.destination);
            for (auto directory = move.destination.parent_path(); !std::filesystem::exists(directory); directory = directory.parent_path()) directories.insert(directory);
        }
        std::size_t completed{};
        try {
            for (const auto& move : moves) {
                tools::files::move(move.source, move.destination);
                ++completed;
            }
        } catch (const std::exception& failure) {
            std::string errors;
            while (completed) {
                const auto& move = moves[--completed];
                try {
                    tools::files::move(move.destination, move.source);
                } catch (const std::exception& error) {
                    errors += "\n" + std::string{error.what()};
                }
            }
            for (const auto& directory : directories | std::views::reverse) {
                try {
                    if (std::filesystem::exists(directory) && std::filesystem::is_empty(directory)) std::filesystem::remove(directory);
                } catch (const std::exception& error) {
                    errors += "\nRemove " + tools::files::utf8(directory) + ": " + error.what();
                }
            }
            if (!errors.empty()) throw std::runtime_error{std::string{failure.what()} + "\nCould not roll back all file moves:" + errors};
            throw;
        }
        return {std::move(moves)};
    }
} // namespace classifier::dataset
