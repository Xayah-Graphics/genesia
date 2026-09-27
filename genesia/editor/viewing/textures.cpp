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
            if (scanned_ready) {
                history       = std::move(scanned);
                scanned_ready = false;
                history_ready = true;
                history_error = std::move(scan_error);
                refreshed     = true;
            }
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
        if (!std::ranges::contains(history, path)) history.push_back(path);
        if (!history_ready) refresh();
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
    void TextureCache::refresh() {
        history_ready = false;
        history_error.clear();
        {
            const std::lock_guard lock{mutex};
            requested.clear();
            results.clear();
            scanned_ready = false;
            rescan        = true;
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
        }
        condition.notify_one();
    }
    void TextureCache::read() {
        for (;;) {
            Load task;
            bool scan{};
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [&] { return closing || rescan || (results.size() < 3 && !requested.empty()); });
                if (closing) break;
                scan = std::exchange(rescan, false);
                if (!scan) task = requested.front();
            }
            if (scan) {
                std::vector<std::filesystem::path> paths;
                std::string failure;
                try {
                    std::filesystem::create_directories(project::output);
                    std::vector<std::pair<std::uint64_t, std::filesystem::path>> images;
                    for (const auto& entry : std::filesystem::directory_iterator{project::output}) {
                        if (!entry.is_regular_file() || entry.path().extension() != ".png") continue;
                        WIN32_FILE_ATTRIBUTE_DATA attributes;
                        if (!GetFileAttributesExW(entry.path().c_str(), GetFileExInfoStandard, &attributes)) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Read image creation time: " + files::utf8(entry.path())};
                        const auto created = (std::uint64_t{attributes.ftCreationTime.dwHighDateTime} << 32) | attributes.ftCreationTime.dwLowDateTime;
                        images.emplace_back(created, entry.path());
                    }
                    std::ranges::sort(images);
                    paths.reserve(images.size());
                    for (auto& [created, path] : images) paths.push_back(std::move(path));
                } catch (const std::exception& error) {
                    failure = error.what();
                }
                {
                    const std::lock_guard lock{mutex};
                    if (closing || rescan) continue;
                    scanned       = std::move(paths);
                    scan_error    = std::move(failure);
                    scanned_ready = true;
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
