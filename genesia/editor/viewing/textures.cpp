module;
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
    void TextureCache::receive() {
        std::vector<Decoded> decoded;
        {
            const std::lock_guard lock{mutex};
            decoded = std::exchange(results, {});
            if (scanned_ready) {
                history.insert(history.end(), scanned.begin(), scanned.end());
                std::ranges::sort(history);
                history.erase(std::unique(history.begin(), history.end()), history.end());
                scanned.clear();
                scanned_ready = false;
                history_ready = true;
                history_error = std::move(scan_error);
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
        condition.notify_one();
    }
    void TextureCache::adopt(const Record& record, std::uint64_t& texture) {
        const auto& path = record.path;
        if (!std::ranges::contains(history, path)) history.push_back(path);
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
        std::vector<std::filesystem::path> paths;
        std::string failure;
        try {
            std::filesystem::create_directories(project::raw);
            for (const auto& entry : std::filesystem::directory_iterator{project::raw})
                if (entry.is_regular_file() && entry.path().extension() == ".png") paths.push_back(entry.path());
        } catch (const std::exception& error) {
            failure = error.what();
        }
        {
            const std::lock_guard lock{mutex};
            scanned       = std::move(paths);
            scan_error    = std::move(failure);
            scanned_ready = true;
            pending       = true;
        }
        glfwPostEmptyEvent();
        for (;;) {
            Load task;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [&] { return closing || (results.size() < 3 && !requested.empty()); });
                if (closing) break;
                task = requested.front();
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
