module;
#include <Windows.h>
#include <GLFW/glfw3.h>
module genesia.editor.viewing.textures;
import genesia.io.files;
import std;
namespace genesia::editor {
    TextureCache::TextureCache(Renderer& display) : renderer{display}, reader{[this] { read(); }} {}
    TextureCache::~TextureCache() {
        shutdown();
        reader.join();
        for (const auto& [path, entry] : entries)
            if (entry.texture) renderer.retire(entry.texture);
    }
    bool TextureCache::receive() {
        std::vector<Decoded> decoded;
        bool refreshed{};
        {
            const std::lock_guard lock{mutex};
            decoded = std::exchange(results, {});
            for (auto& result : scanned) {
                for (auto& [path, folder] : result.folders) folders[path] = std::move(folder);
                if (result.path != directory) continue;
                history       = std::move(result.images);
                history_ready = true;
                history_error = std::move(result.error);
                refreshed     = true;
            }
            scanned.clear();
            pending = false;
        }
        for (auto& result : decoded) {
            auto& cached   = entries[result.path];
            cached.touched = ++clock;
            cached.checked = std::chrono::steady_clock::now();
            if (result.unchanged) continue;
            if (cached.texture) renderer.retire(std::exchange(cached.texture, 0));
            texture_bytes -= cached.bytes;
            cached.bytes        = 0;
            cached.modified     = result.modified;
            cached.record       = std::move(result.record);
            cached.record_error = std::move(result.record_error);
            cached.error        = std::move(result.error);
            if (cached.error.empty()) {
                cached.width   = result.image.width;
                cached.height  = result.image.height;
                cached.texture = renderer.upload(result.image);
                cached.bytes   = std::size_t(cached.width) * cached.height * 4;
                texture_bytes += cached.bytes;
            }
        }
        if (refreshed) for (auto& [path, cached] : entries) cached.checked = {};
        condition.notify_one();
        return refreshed;
    }
    void TextureCache::adopt(const Record& record, std::uint64_t& texture) {
        const auto& path = record.path;
        const auto relative = path.lexically_relative(directory);
        if (!relative.empty() && *relative.begin() != ".." && !std::ranges::contains(history, path)) history.push_back(path);
        if (!history_ready) refresh(directory);
        if (!texture) return;
        auto& cached    = entries[path];
        cached.record   = record;
        cached.width    = record.parameters.width;
        cached.height   = record.parameters.height;
        cached.modified = std::filesystem::last_write_time(path);
        cached.checked  = std::chrono::steady_clock::now();
        cached.texture  = std::exchange(texture, 0);
        cached.touched  = ++clock;
        cached.bytes    = std::size_t(cached.width) * cached.height * 4;
        texture_bytes += cached.bytes;
        {
            const std::lock_guard lock{mutex};
            std::erase_if(requested, [&](const Load& load) { return load.path == path; });
            std::erase_if(results, [&](const Decoded& result) { return result.path == path; });
        }
        condition.notify_one();
    }
    void TextureCache::refresh(const std::filesystem::path& path) {
        if (directory != path) history.clear();
        directory = path;
        history_ready = false;
        history_error.clear();
        {
            const std::lock_guard lock{mutex};
            requested.clear();
            results.clear();
        }
        scan(path);
    }
    void TextureCache::scan(const std::filesystem::path& path) {
        folders[path].ready = false;
        {
            const std::lock_guard lock{mutex};
            std::erase(requested_directories, path);
            std::erase_if(scanned, [&](const auto& result) { return result.path == path; });
            requested_directories.push_back(path);
        }
        condition.notify_one();
    }
    void TextureCache::request(std::vector<std::filesystem::path> files) {
        std::set<std::filesystem::path> pinned;
        std::vector<Load> missing;
        const auto now = std::chrono::steady_clock::now();
        for (const auto& path : files) {
            if (!pinned.insert(path).second) continue;
            const auto cached = entries.find(path);
            if (cached == entries.end()) missing.push_back({path});
            else {
                cached->second.touched = ++clock;
                if (now - cached->second.checked > std::chrono::seconds{2}) missing.push_back({path, cached->second.modified});
            }
        }
        while (texture_bytes > 512ULL * 1024 * 1024 || entries.size() > 128) {
            auto oldest = entries.end();
            for (auto entry = entries.begin(); entry != entries.end(); ++entry)
                if (!pinned.contains(entry->first) && (oldest == entries.end() || entry->second.touched < oldest->second.touched)) oldest = entry;
            if (oldest == entries.end()) break;
            if (oldest->second.texture) renderer.retire(oldest->second.texture);
            texture_bytes -= oldest->second.bytes;
            entries.erase(oldest);
        }
        {
            const std::lock_guard lock{mutex};
            if (closing) return;
            std::erase_if(missing, [&](const Load& load) { return std::ranges::any_of(results, [&](const Decoded& result) { return result.path == load.path; }); });
            requested = std::move(missing);
        }
        condition.notify_one();
    }
    void TextureCache::discard(const std::filesystem::path& path) {
        const auto found = entries.find(path);
        if (found != entries.end()) {
            if (found->second.texture) renderer.retire(found->second.texture);
            texture_bytes -= found->second.bytes;
            entries.erase(found);
        }
        std::erase(history, path);
        {
            const std::lock_guard lock{mutex};
            std::erase_if(requested, [&](const Load& load) { return load.path == path; });
            std::erase_if(results, [&](const Decoded& result) { return result.path == path; });
        }
    }
    void TextureCache::shutdown() {
        {
            const std::lock_guard lock{mutex};
            closing = true;
            requested.clear();
            requested_directories.clear();
        }
        condition.notify_one();
    }
    void TextureCache::read() {
        for (;;) {
            Load task;
            std::optional<std::filesystem::path> directory;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [&] { return closing || !requested_directories.empty() || (results.size() < 3 && !requested.empty()); });
                if (closing) break;
                if (!requested_directories.empty()) {
                    directory = std::move(requested_directories.front());
                    requested_directories.pop_front();
                } else task = requested.front();
            }
            if (directory) {
                Directory result{.path = *directory};
                result.folders[*directory];
                try {
                    if (*directory == project::output) std::filesystem::create_directories(*directory);
                    std::vector<std::pair<std::uint64_t, std::filesystem::path>> images;
                    for (const auto& entry : std::filesystem::recursive_directory_iterator{*directory}) {
                        if (entry.is_directory()) {
                            result.folders[entry.path().parent_path()].children.push_back(entry.path());
                            result.folders[entry.path()];
                            continue;
                        }
                        if (!entry.is_regular_file() || (entry.path().extension() != ".png" && entry.path().extension() != ".PNG")) continue;
                        WIN32_FILE_ATTRIBUTE_DATA attributes;
                        if (!GetFileAttributesExW(entry.path().c_str(), GetFileExInfoStandard, &attributes)) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Read image creation time: " + files::utf8(entry.path())};
                        const auto created = (std::uint64_t{attributes.ftCreationTime.dwHighDateTime} << 32) | attributes.ftCreationTime.dwLowDateTime;
                        images.emplace_back(created, entry.path());
                        ++result.folders[entry.path().parent_path()].images;
                    }
                    for (auto& [path, folder] : result.folders | std::views::reverse) {
                        std::ranges::sort(folder.children);
                        folder.ready = true;
                        if (path != *directory) result.folders.at(path.parent_path()).images += folder.images;
                    }
                    std::ranges::sort(images);
                    result.images.reserve(images.size());
                    for (auto& [created, path] : images) result.images.push_back(std::move(path));
                } catch (const std::exception& error) {
                    result.error = error.what();
                    result.folders[*directory].error = result.error;
                    result.folders[*directory].ready = true;
                }
                {
                    const std::lock_guard lock{mutex};
                    if (closing || std::ranges::contains(requested_directories, *directory)) continue;
                    scanned.push_back(std::move(result));
                    pending       = true;
                }
                glfwPostEmptyEvent();
                continue;
            }
            Decoded result{.path = task.path};
            try {
                result.modified  = std::filesystem::last_write_time(task.path);
                result.unchanged = task.modified == result.modified;
                if (!result.unchanged) {
                    result.image = read_image(task.path);
                    try {
                        result.record = read_record(task.path);
                    } catch (const std::exception& error) {
                        result.record_error = error.what();
                    }
                }
            } catch (const std::exception& error) {
                result.error = std::format("{}: {}", files::utf8(task.path), error.what());
            }
            {
                const std::lock_guard lock{mutex};
                if (!std::ranges::contains(requested, task)) continue;
                std::erase(requested, task);
                results.push_back(std::move(result));
                pending = true;
            }
            glfwPostEmptyEvent();
        }
    }
} // namespace genesia::editor
