module;
#if defined(_WIN32)
#include <Windows.h>
#endif
module genesia.data.operations;
import genesia.data.datasets;
import genesia.io.files;
import std;
namespace genesia::dataset {
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
            if (!reserved.insert(move.destination).second || (std::filesystem::exists(move.destination) && !vacated.contains(move.destination))) throw std::runtime_error{"Destination already exists: " + files::utf8(move.destination)};
            vacated.insert(move.source);
            vacated.erase(move.destination);
            for (auto directory = move.destination.parent_path(); !std::filesystem::exists(directory); directory = directory.parent_path()) directories.insert(directory);
        }
        std::size_t completed{};
        try {
            for (const auto& move : moves) {
                files::move(move.source, move.destination);
                ++completed;
            }
        } catch (const std::exception& failure) {
            std::string errors;
            while (completed) {
                const auto& move = moves[--completed];
                try {
                    files::move(move.destination, move.source);
                } catch (const std::exception& error) {
                    errors += "\n" + std::string{error.what()};
                }
            }
            for (const auto& directory : directories | std::views::reverse) {
                try {
                    if (std::filesystem::exists(directory) && std::filesystem::is_empty(directory)) std::filesystem::remove(directory);
                } catch (const std::exception& error) {
                    errors += "\nRemove " + files::utf8(directory) + ": " + error.what();
                }
            }
            if (!errors.empty()) throw std::runtime_error{std::string{failure.what()} + "\nCould not roll back all file moves:" + errors};
            throw;
        }
        return {std::move(moves)};
    }
    DeleteResult delete_image(Index& index, const std::string_view name, const std::string_view sha) {
        const auto found = std::ranges::find(index.roots, name, [](const Root& root) { return root.all.key; });
        if (found == index.roots.end() || !found->ready) throw std::runtime_error{"Dataset is not ready: " + std::string{name}};
        auto& root = *found;
        DeleteResult result{std::string{name}, std::string{sha}};
        std::vector<std::filesystem::path> paths;
        for (const auto& file : root.files)
            if (file.sha == sha) paths.push_back(file.path);
        if (paths.empty()) throw std::runtime_error{"Image is not in dataset: " + std::string{name} + " / " + std::string{sha}};
        for (const auto& path : paths) {
            std::error_code error;
            const bool removed = std::filesystem::remove(path, error);
            if (error || !removed) {
                result.error = std::format("Permanently deleted {} of {} file entries.\nDelete {}: {}", result.paths.size(), paths.size(), files::utf8(path), error ? error.message() : "File no longer exists");
                break;
            }
            result.paths.push_back(path);
        }
        if (result.paths.empty()) return result;
        std::erase_if(root.files, [&](const File& file) { return std::ranges::contains(result.paths, file.path); });
        index.rebuild(root);
        return result;
    }
    NormalizeResult normalize(Index& index, const std::string_view name, const std::atomic_bool& interrupted, const std::function<void(NormalizeStage, std::size_t, std::size_t)>& progress) {
        index.scan(name);
        auto& root = *std::ranges::find(index.roots, name, [](const Root& root) { return root.all.key; });
        NormalizeResult result{.root = std::string{name}, .files = root.files.size(), .error = root.error};
        if (!result.error.empty()) return result;
        std::map<std::string, const File*> resources;
        std::vector<std::pair<File*, const File*>> replacements;
        for (auto& file : root.files) {
            const auto [source, added] = resources.emplace(file.sha, &file);
            if (!added && file.entity != source->second->entity) replacements.emplace_back(&file, source->second);
        }
        progress(NormalizeStage::linking, 0, replacements.size());
        for (const auto [target, source] : replacements) {
            if (interrupted.load()) {
                result.stopped = true;
                break;
            }
            try {
                auto linked    = *source;
                linked.path    = target->path;
                auto temporary = target->path;
                temporary += ".genesia-link";
                std::filesystem::create_hard_link(source->path, temporary);
                try {
                    files::publish(temporary, target->path);
                } catch (const std::exception& failure) {
                    std::error_code cleanup;
                    std::filesystem::remove(temporary, cleanup);
                    if (cleanup) throw std::runtime_error{std::format("{}\nRemove {}: {}", failure.what(), files::utf8(temporary), cleanup.message())};
                    throw;
                }
                *target = std::move(linked);
                ++result.linked;
            } catch (const std::exception& failure) {
                result.error = failure.what();
                break;
            }
            if (result.linked % 64 == 0) progress(NormalizeStage::linking, result.linked, replacements.size());
        }
        index.rebuild(root);
        progress(NormalizeStage::linking, result.linked, replacements.size());
        if (!result.error.empty() || result.stopped) return result;
        if (interrupted.load()) {
            result.stopped = true;
            return result;
        }
        std::map<std::filesystem::path, std::size_t> numbers;
        std::vector<Move> renamed, staged, finished;
        for (const auto& file : root.files) {
            const auto directory   = file.path.parent_path();
            const auto destination = directory / std::format("{:05}.png", ++numbers[directory]);
            if (file.path == destination) continue;
            auto temporary = file.path;
            temporary += ".genesia-rename";
            renamed.push_back({file.path, destination, file.sha});
            staged.push_back({file.path, temporary, file.sha});
            finished.push_back({temporary, destination, file.sha});
        }
        if (renamed.empty()) return result;
        const auto count = renamed.size();
        if (interrupted.load()) {
            result.stopped = true;
            return result;
        }
        progress(NormalizeStage::renaming, 0, count);
        try {
            std::ranges::move(finished, std::back_inserter(staged));
            move_images(std::move(staged));
            result.renamed = std::move(renamed);
            index.apply(result.renamed);
        } catch (const std::exception& failure) {
            result.error = failure.what();
            root.error   = result.error;
            root.ready   = false;
        }
        progress(NormalizeStage::renaming, result.renamed.size(), count);
        return result;
    }
} // namespace genesia::dataset
