module;
#include <GLFW/glfw3.h>
module genesia.editor.viewing.textures;
import std;
namespace genesia::editor {
    TextureCache::TextureCache(Renderer& display) : renderer{display}, worker{[this] { read(); }} {}
    TextureCache::~TextureCache() {
        {
            const std::lock_guard lock{mutex};
            closing = true;
        }
        condition.notify_all();
        worker.join();
        for (const auto& [sha, entry] : entries)
            if (entry.texture) renderer.retire(entry.texture);
    }
    void TextureCache::receive() {
        std::vector<Decoded> decoded;
        {
            const std::lock_guard lock{mutex};
            decoded = std::exchange(results, {});
            pending = false;
        }
        for (auto& result : decoded) {
            auto& cached = entries[result.file.sha];
            if (cached.texture) continue;
            texture_bytes -= cached.bytes;
            cached = {std::move(result.file), std::move(result.record), 0, ++clock, 0, std::move(result.error)};
            if (cached.error.empty()) {
                cached.texture = renderer.upload(result.image);
                cached.bytes   = static_cast<std::size_t>(result.image.width) * result.image.height * 4;
                texture_bytes += cached.bytes;
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
            texture_bytes -= cached.bytes;
            cached = {file, record, std::exchange(texture, 0), ++clock, bytes, {}};
            texture_bytes += bytes;
        }
        {
            const std::lock_guard lock{mutex};
            std::erase_if(requested, [&](const auto& pending) { return pending.sha == file.sha; });
            std::erase_if(results, [&](const auto& pending) { return pending.file.sha == file.sha; });
            pending = !results.empty();
        }
        condition.notify_all();
    }

    void TextureCache::request(std::vector<dataset::File> files) {
        std::set<std::string> pinned;
        std::vector<dataset::File> missing;
        for (const auto& file : files) {
            if (!pinned.insert(file.sha).second) continue;
            const auto cached = entries.find(file.sha);
            if (cached != entries.end()) cached->second.touched = ++clock;
            else if (!paused) missing.push_back(file);
        }
        while (texture_bytes > 512ULL * 1024 * 1024) {
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
            if (requested == missing) return;
            requested = std::move(missing);
        }
        condition.notify_all();
    }

    void TextureCache::discard(const std::span<const std::filesystem::path> paths) {
        std::erase_if(entries, [&](const auto& item) {
            const auto& cached = item.second;
            if (!std::ranges::contains(paths, cached.file.path)) return false;
            if (cached.texture) renderer.retire(cached.texture);
            texture_bytes -= cached.bytes;
            return true;
        });
        {
            const std::lock_guard lock{mutex};
            std::erase_if(requested, [&](const auto& file) { return std::ranges::contains(paths, file.path); });
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
                cached.file.path = cached.record.path = found->second->destination;
                if (!cached.error.empty()) {
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
        std::vector<dataset::File> delivered;
        for (;;) {
            dataset::File task;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [&] {
                    if (closing) return true;
                    std::erase_if(delivered, [&](const auto& file) { return !std::ranges::contains(requested, file); });
                    return results.size() < 2 && std::ranges::any_of(requested, [&](const auto& file) { return !std::ranges::contains(delivered, file); });
                });
                if (closing) break;
                task = *std::ranges::find_if(requested, [&](const auto& file) { return !std::ranges::contains(delivered, file); });
            }
            Decoded result{task};
            try {
                result.record = read_record(task.path);
                result.image  = read_image(task.path);
            } catch (const std::exception& error) {
                result.error = std::format("{}: {}", task.path.string(), error.what());
            }
            {
                const std::lock_guard lock{mutex};
                if (!std::ranges::contains(requested, task)) continue;
                delivered.push_back(task);
                results.push_back(std::move(result));
                pending = true;
            }
            glfwPostEmptyEvent();
        }
    }
} // namespace genesia::editor
