module;
#include <GLFW/glfw3.h>
module genesia.editor.prompt.previews;
import genesia.io.files;
import std;

namespace genesia::editor::previews {
    namespace {
        void prune(std::filesystem::path directory, const std::filesystem::path& root) {
            for (; directory != root.parent_path(); directory = directory.parent_path()) {
                if (!std::filesystem::exists(directory)) continue;
                if (!std::filesystem::is_empty(directory)) break;
                std::filesystem::remove(directory);
            }
        }
    } // namespace

    std::filesystem::path long_path(std::filesystem::path path) {
        path             = std::filesystem::absolute(path).lexically_normal().make_preferred();
        const auto& text = path.native();
        if (text.starts_with(L"\\\\?\\")) return path;
        return text.starts_with(L"\\\\") ? L"\\\\?\\UNC\\" + text.substr(2) : L"\\\\?\\" + text;
    }

    Location::Location(std::filesystem::path base, const std::map<std::string, std::string>& parts) : root{long_path(std::move(base))}, directory{root} {
        for (const auto& [part, option] : parts) directory /= files::path(part + "=" + option);
    }

    std::filesystem::path Location::scene(const std::string_view name, const std::map<std::string, std::string>& variations) const {
        auto path = directory / "scenes" / files::path(name);
        for (const auto& [group, option] : variations) path /= files::path(group + "=" + option);
        return path / "preview.png";
    }

    Images::Images(Renderer& display) : renderer{display}, worker{[this] { read(); }} {}

    Images::~Images() {
        {
            const std::lock_guard lock{mutex};
            closing = true;
            std::erase_if(requested, [](const auto& request) { return request.operation == Operation::read; });
        }
        condition.notify_one();
        worker.join();
        for (const auto& [path, texture] : textures)
            if (texture.id) renderer.retire(texture.id);
    }

    void Images::update(const std::set<std::filesystem::path>& wanted) {
        std::vector<Result> results;
        {
            const std::lock_guard lock{mutex};
            results = std::exchange(completed, {});
            pending = false;
            std::erase_if(requested, [&](const auto& request) { return request.operation == Operation::read && !wanted.contains(request.path); });
        }
        std::erase_if(textures, [&](const auto& entry) {
            if (wanted.contains(entry.first)) return false;
            if (entry.second.id) renderer.retire(entry.second.id);
            return true;
        });
        for (auto& result : results) {
            if (result.operation != Operation::read) {
                saving = false;
                error  = std::move(result.error);
                if (!result.applied) continue;
                status = result.operation == Operation::clear ? "Preview cleared" : "Preview saved";
            }
            if (!wanted.contains(result.path)) continue;
            auto& texture = textures[result.path];
            if (texture.id) renderer.retire(texture.id);
            texture = {.loading = false, .exists = result.exists, .error = std::move(result.image_error)};
            if (!result.image.pixels.empty()) {
                texture.width  = result.image.width;
                texture.height = result.image.height;
                texture.id     = renderer.upload(result.image);
            }
        }
        for (const auto& path : wanted) request(path);
    }

    const Images::Texture& Images::request(const std::filesystem::path& path) {
        const auto [entry, inserted] = textures.try_emplace(path);
        if (inserted) {
            {
                const std::lock_guard lock{mutex};
                requested.push_back({.path = path});
            }
            condition.notify_one();
        }
        return entry->second;
    }

    void Images::edit(std::filesystem::path path, std::filesystem::path root, std::optional<std::filesystem::path> source) {
        Request request{.operation = source ? Operation::replace : Operation::clear, .path = std::move(path), .root = std::move(root)};
        if (source) request.source = long_path(std::move(*source));
        {
            const std::lock_guard lock{mutex};
            requested.push_back(std::move(request));
        }
        saving = true;
        error.clear();
        status.clear();
        condition.notify_one();
    }

    void Images::read() {
        for (;;) {
            Request request;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [&] { return closing || !requested.empty(); });
                if (requested.empty()) return;
                request = std::move(requested.front());
                requested.pop_front();
            }
            Result result{.operation = request.operation, .path = request.path};
            auto temporary = request.path;
            temporary += ".part";
            bool candidate{};
            try {
                if (request.operation != Operation::read) {
                    if (request.operation == Operation::replace) {
                        const auto bytes = files::read_bytes(request.source);
                        candidate        = true;
                        std::filesystem::create_directories(request.path.parent_path());
                        std::ofstream file{temporary, std::ios::binary | std::ios::trunc};
                        file.exceptions(std::ios::badbit | std::ios::failbit);
                        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                        file.close();
                        // Decode before publishing: a failed drop must leave the previous file intact.
                        result.image = read_image(temporary);
                        files::publish(temporary, request.path);
                        candidate = false;
                    } else std::filesystem::remove(request.path);
                    result.applied = true;
                    if (request.operation == Operation::clear) prune(request.path.parent_path(), request.root);
                }
            } catch (const std::exception& failure) {
                result.error = std::format("{}: {}", files::utf8(request.path), failure.what());
                if (candidate) {
                    try {
                        std::filesystem::remove(temporary);
                        prune(request.path.parent_path(), request.root);
                    } catch (const std::exception& cleanup) {
                        result.error += std::format("\nPreview cleanup: {}", cleanup.what());
                    }
                }
            }
            if (request.operation == Operation::read || result.applied) {
                try {
                    result.exists = std::filesystem::exists(request.path);
                    if (result.exists && result.image.pixels.empty()) result.image = read_image(request.path);
                } catch (const std::exception& failure) {
                    result.image_error = std::format("{}: {}", files::utf8(request.path), failure.what());
                }
            }
            {
                const std::lock_guard lock{mutex};
                completed.push_back(std::move(result));
                pending = true;
            }
            glfwPostEmptyEvent();
        }
    }
} // namespace genesia::editor::previews
