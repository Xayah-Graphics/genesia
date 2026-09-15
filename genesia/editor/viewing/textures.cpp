module;
#include <GLFW/glfw3.h>
module genesia.editor.viewing.textures;
import std;
import vulkan;
namespace genesia::editor {
    TextureCache::TextureCache(Renderer& display) : renderer{display}, worker{[this] { read(); }} {}
    TextureCache::~TextureCache() {
        {
            const std::lock_guard lock{mutex};
            closing = true;
        }
        condition.notify_all();
        worker.join();
        for (const auto& [sha, entry] : entries) {
            if (entry.texture) renderer.retire(entry.texture);
            if (entry.mask) renderer.retire(entry.mask);
        }
    }
    void TextureCache::receive() {
        std::vector<Decoded> decoded;
        {
            const std::lock_guard lock{mutex};
            decoded = std::exchange(results, {});
            pending = false;
        }
        for (auto& result : decoded) {
            auto& cached   = entries[result.file.sha];
            cached.file    = std::move(result.file);
            cached.touched = ++clock;
            if (result.mask) {
                cached.mask_checked = true;
                cached.mask_error   = std::move(result.error);
                if (!cached.mask && !result.image.pixels.empty()) {
                    cached.mask = renderer.texture({std::uint32_t(result.image.width), std::uint32_t(result.image.height)}, vk::Format::eR8Unorm);
                    renderer.upload(cached.mask, result.image.pixels.data(), result.image.width, result.image.height, true);
                    const auto bytes = result.image.pixels.size();
                    cached.bytes += bytes;
                    texture_bytes += bytes;
                }
            } else if (!cached.texture) {
                cached.record = std::move(result.record);
                cached.error  = std::move(result.error);
                if (cached.error.empty()) {
                    cached.texture   = renderer.upload(result.image);
                    const auto bytes = static_cast<std::size_t>(result.image.width) * result.image.height * 4;
                    cached.bytes += bytes;
                    texture_bytes += bytes;
                }
            }
        }
        condition.notify_all();
    }

    void TextureCache::adopt(const dataset::File& file, const Record& record, std::uint64_t& texture) {
        auto& cached = entries[file.sha];
        if (cached.texture) {
            renderer.retire(std::exchange(texture, 0));
            cached.touched = ++clock;
        } else {
            const auto bytes = static_cast<std::size_t>(file.width) * file.height * 4;
            cached.file      = file;
            cached.record    = record;
            cached.texture   = std::exchange(texture, 0);
            cached.touched   = ++clock;
            cached.bytes += bytes;
            cached.error.clear();
            texture_bytes += bytes;
        }
        {
            const std::lock_guard lock{mutex};
            std::erase_if(requested, [&](const auto& pending) { return !pending.mask && pending.file.sha == file.sha; });
            std::erase_if(results, [&](const auto& pending) { return !pending.mask && pending.file.sha == file.sha; });
            pending = !results.empty();
        }
        condition.notify_all();
    }

    void TextureCache::request(std::vector<dataset::File> files, const bool masks) {
        std::set<std::string> pinned;
        std::vector<Load> missing;
        for (const auto& file : files) {
            if (!pinned.insert(file.sha).second) continue;
            const auto cached = entries.find(file.sha);
            if (cached != entries.end()) cached->second.touched = ++clock;
            if (!paused && (cached == entries.end() || (!cached->second.texture && cached->second.error.empty()))) missing.push_back({file});
            if (!paused && masks && (cached == entries.end() || !cached->second.mask_checked)) missing.push_back({file, true});
        }
        while (texture_bytes > 512ULL * 1024 * 1024) {
            auto oldest = entries.end();
            for (auto entry = entries.begin(); entry != entries.end(); ++entry)
                if (!pinned.contains(entry->first) && (oldest == entries.end() || entry->second.touched < oldest->second.touched)) oldest = entry;
            if (oldest == entries.end()) break;
            if (oldest->second.texture) renderer.retire(oldest->second.texture);
            if (oldest->second.mask) renderer.retire(oldest->second.mask);
            texture_bytes -= oldest->second.bytes;
            entries.erase(oldest);
        }
        {
            const std::lock_guard lock{mutex};
            std::erase_if(missing, [&](const Load& load) { return std::ranges::any_of(results, [&](const Decoded& result) { return result.file == load.file && result.mask == load.mask; }); });
            if (requested == missing) return;
            requested = std::move(missing);
        }
        condition.notify_all();
    }

    void TextureCache::mask_ready(const std::string& sha) {
        const auto entry = entries.find(sha);
        if (entry != entries.end() && !entry->second.mask) entry->second.mask_checked = false;
        const std::lock_guard lock{mutex};
        std::erase_if(requested, [&](const auto& load) { return load.mask && load.file.sha == sha; });
        std::erase_if(results, [&](const auto& result) { return result.mask && result.file.sha == sha; });
    }

    void TextureCache::discard(const std::span<const std::filesystem::path> paths) {
        std::erase_if(entries, [&](const auto& item) {
            const auto& cached = item.second;
            if (!std::ranges::contains(paths, cached.file.path)) return false;
            if (cached.texture) renderer.retire(cached.texture);
            if (cached.mask) renderer.retire(cached.mask);
            texture_bytes -= cached.bytes;
            return true;
        });
        {
            const std::lock_guard lock{mutex};
            std::erase_if(requested, [&](const auto& file) { return std::ranges::contains(paths, file.file.path); });
            std::erase_if(results, [&](const auto& result) { return std::ranges::contains(paths, result.file.path); });
            pending = !results.empty();
        }
        condition.notify_all();
    }

    void TextureCache::relocate(const std::span<const dataset::Move> moves) {
        std::map<std::filesystem::path, const dataset::Move*> destinations;
        for (const auto& move : moves) destinations.emplace(move.source, &move);
        for (auto entry = entries.begin(); entry != entries.end();) {
            auto& cached     = entry->second;
            const auto found = destinations.find(cached.file.path);
            if (found != destinations.end() && entry->first == found->second->sha) {
                cached.file.path = found->second->destination;
                if (cached.record) cached.record->path = cached.file.path;
                if (!cached.error.empty()) {
                    if (cached.mask) renderer.retire(cached.mask);
                    texture_bytes -= cached.bytes;
                    entry = entries.erase(entry);
                    continue;
                }
            }
            ++entry;
        }
        {
            const std::lock_guard lock{mutex};
            requested.clear();
            results.clear();
            pending = false;
        }
        condition.notify_all();
    }

    void TextureCache::read() {
        for (;;) {
            Load task;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [&] { return closing || (results.size() < 2 && !requested.empty()); });
                if (closing) break;
                task = requested.front();
            }
            Decoded result{.file = task.file, .mask = task.mask};
            try {
                if (task.mask) {
                    if (auto cached = foreground::read_cached(task.file)) result.image = std::move(*cached);
                } else {
                    result.record = read_image_info(task.file.path).record;
                    result.image  = read_image(task.file.path);
                }
            } catch (const std::exception& error) {
                result.error = std::format("{}: {}", task.file.path.string(), error.what());
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
