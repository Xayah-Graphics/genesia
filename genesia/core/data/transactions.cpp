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
            if (file.sha != entry->at("sha").get<std::string>() || file.entity != entry->at("entity").get<std::string>()) throw std::runtime_error{"Cannot recover changed image: " + files::utf8(file.path)};
            if (at_destination) {
                files::move(destination, source);
                result.paths.push_back({destination, source, file.sha, file.entity});
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
            operation["moves"].push_back({{"source", files::utf8(move.source)}, {"destination", files::utf8(move.destination)}, {"sha", move.sha}, {"entity", move.entity}});
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
            paths.push_back({current, original, move.at("sha"), move.at("entity")});
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
} // namespace genesia::dataset
