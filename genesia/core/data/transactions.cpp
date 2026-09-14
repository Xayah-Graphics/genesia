module;
#include <nlohmann/json.hpp>
#if defined(_WIN32)
#include <Windows.h>
#endif
module genesia.data.transactions;
import genesia.project;
import genesia.data.datasets;
import genesia.io.files;
import std;
namespace genesia::dataset {
    MoveResult recover_moves(const std::filesystem::path& journal) {
        if (!std::filesystem::exists(journal)) return {};
        auto state = files::read_json(journal);
        if (state.at("version") != 1) throw std::runtime_error{"Unsupported image move journal"};
        auto& history      = state.at("history");
        const auto pending = std::ranges::find_if(history, [](const auto& item) { return item.at("status") == "applying" || item.at("status") == "undoing"; });
        if (pending == history.end()) return {};
        MoveResult result;
        Index index;
        const bool undoing = pending->at("status") == "undoing";
        auto& moves        = pending->at("moves");
        for (auto entry = moves.rbegin(); entry != moves.rend(); ++entry) {
            const auto source      = files::path(entry->at(undoing ? "destination" : "source").get<std::string>());
            const auto destination = files::path(entry->at(undoing ? "source" : "destination").get<std::string>());
            const bool at_source = std::filesystem::exists(source), at_destination = std::filesystem::exists(destination);
            if (at_source == at_destination) throw std::runtime_error{"Cannot recover image move: " + files::utf8(source) + " -> " + files::utf8(destination)};
            const auto file = index.identify(at_source ? source : destination);
            if (file.sha != entry->at("sha").get<std::string>()) throw std::runtime_error{"Cannot recover changed image: " + files::utf8(file.path)};
            if (at_destination) {
                files::move(destination, source);
                result.paths.push_back({destination, source, file.sha});
                ++result.restored;
            }
        }
        if (!undoing)
            for (const auto& directory : pending->at("created")) {
                const auto path = files::path(directory.get<std::string>());
                if (std::filesystem::exists(path) && std::filesystem::is_empty(path)) std::filesystem::remove(path);
            }
        (*pending)["status"] = undoing ? "applied" : "rolled_back";
        files::write_json(journal, state);
        return result;
    }
    MoveResult move_images(std::vector<Move> moves, const std::filesystem::path& journal) {
        recover_moves(journal);
        const auto ordered = [](const std::filesystem::path& a, const std::filesystem::path& b) {
#if defined(_WIN32)
            return CompareStringOrdinal(a.c_str(), -1, b.c_str(), -1, TRUE) == CSTR_LESS_THAN;
#else
            return a < b;
#endif
        };
        std::set<std::filesystem::path, decltype(ordered)> reserved;
        std::set<std::filesystem::path> directories;
        auto state = std::filesystem::exists(journal) ? files::read_json(journal) : nlohmann::json{{"version", 1}, {"history", nlohmann::json::array()}};
        nlohmann::json operation{{"status", "applying"}, {"moves", nlohmann::json::array()}, {"created", nlohmann::json::array()}};
        for (const auto& move : moves) {
            if (!reserved.insert(move.destination).second || std::filesystem::exists(move.destination)) throw std::runtime_error{"Destination already exists: " + files::utf8(move.destination)};
            operation["moves"].push_back({{"source", files::utf8(move.source)}, {"destination", files::utf8(move.destination)}, {"sha", move.sha}});
            for (auto directory = move.destination.parent_path(); !std::filesystem::exists(directory); directory = directory.parent_path()) directories.insert(directory);
        }
        for (auto path = directories.rbegin(); path != directories.rend(); ++path) operation["created"].push_back(files::utf8(*path));
        state["history"].push_back(std::move(operation));
        files::write_json(journal, state);
        try {
            for (const auto& move : moves) files::move(move.source, move.destination);
            state["history"].back()["status"] = "applied";
            files::write_json(journal, state);
        } catch (...) {
            recover_moves(journal);
            throw;
        }
        return {.moved = moves.size(), .paths = std::move(moves)};
    }
    MoveResult undo_moves(const std::filesystem::path& journal) {
        recover_moves(journal);
        auto state           = files::read_json(journal);
        auto& history        = state.at("history");
        const auto operation = std::ranges::find_if(history | std::views::reverse, [](const auto& item) { return item.at("status") == "applied"; });
        if (operation == history.rend()) throw std::runtime_error{"No image move to undo"};
        std::vector<Move> paths;
        for (const auto& move : operation->at("moves")) {
            const auto original = files::path(move.at("source").get<std::string>());
            const auto current  = files::path(move.at("destination").get<std::string>());
            if (std::filesystem::exists(original)) throw std::runtime_error{"Undo destination already exists: " + files::utf8(original)};
            paths.push_back({current, original, move.at("sha")});
        }
        (*operation)["status"] = "undoing";
        files::write_json(journal, state);
        try {
            const auto& moves = operation->at("moves");
            for (auto move = moves.rbegin(); move != moves.rend(); ++move) files::move(files::path(move->at("destination").get<std::string>()), files::path(move->at("source").get<std::string>()));
            (*operation)["status"] = "undone";
            files::write_json(journal, state);
        } catch (...) {
            recover_moves(journal);
            throw;
        }
        return {.restored = paths.size(), .paths = std::move(paths)};
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
        std::set<std::filesystem::path> journals;
        const auto folder = project::directory / files::path(name);
        for (const auto& path : result.paths) {
            const auto relative = path.lexically_relative(folder);
            if (std::distance(relative.begin(), relative.end()) > 1) journals.insert(folder / *relative.begin() / ".genesia" / "audit-moves.json");
        }
        for (const auto& journal : journals) {
            try {
                if (!std::filesystem::exists(journal)) continue;
                auto state         = files::read_json(journal);
                auto& history      = state.at("history").get_ref<nlohmann::json::array_t&>();
                const auto removed = std::erase_if(history, [&](const auto& operation) { return std::ranges::any_of(operation.at("moves"), [&](const nlohmann::json& move) { return move.at("sha").get_ref<const std::string&>() == sha; }); });
                if (removed) files::write_json(journal, state);
            } catch (const std::exception& error) {
                if (!result.error.empty()) result.error += '\n';
                result.error += "Images were deleted, but audit undo history could not be updated:\n" + files::utf8(journal) + ": " + error.what();
            }
        }
        return result;
    }
    void recover_normalization(const std::filesystem::path& root) {
        const auto journal = root / ".genesia" / "normalize.json";
        if (!std::filesystem::exists(journal)) return;
        recover_moves(journal);
        const auto state    = files::read_json(journal);
        const auto& history = state.at("history");
        if (history.size() == 2 && history.back().at("status") == "applied") {
            std::set<std::filesystem::path> audits;
            for (const auto& move : history.back().at("moves")) {
                const auto relative = files::path(move.at("destination").get<std::string>()).lexically_relative(root);
                if (std::distance(relative.begin(), relative.end()) > 1) audits.insert(root / *relative.begin() / ".genesia" / "audit-moves.json");
            }
            for (const auto& audit : audits)
                if (std::filesystem::remove(audit) && std::filesystem::is_empty(audit.parent_path())) std::filesystem::remove(audit.parent_path());
        } else if (history.front().at("status") == "applied") undo_moves(journal);
        std::filesystem::remove(journal);
        if (std::filesystem::is_empty(journal.parent_path())) std::filesystem::remove(journal.parent_path());
    }
    NormalizeResult normalize(Index& index, const std::string_view name, const std::atomic_bool& interrupted, const std::function<void(NormalizeStage, std::size_t, std::size_t)>& progress) {
        const auto folder = project::directory / files::path(name);
        recover_normalization(folder);
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
        const auto journal = folder / ".genesia" / "normalize.json";
        const auto count   = renamed.size();
        if (interrupted.load()) {
            result.stopped = true;
            return result;
        }
        progress(NormalizeStage::renaming, 0, count);
        try {
            move_images(std::move(staged), journal);
            move_images(std::move(finished), journal);
            result.renamed = std::move(renamed);
            index.apply(result.renamed);
        } catch (const std::exception& failure) {
            result.error = failure.what();
        }
        try {
            recover_normalization(folder);
        } catch (const std::exception& failure) {
            if (!result.error.empty()) result.error += '\n';
            result.error += failure.what();
            root.error = "Normalization recovery failed: " + result.error;
            root.ready = false;
        }
        progress(NormalizeStage::renaming, result.renamed.size(), count);
        return result;
    }
} // namespace genesia::dataset
