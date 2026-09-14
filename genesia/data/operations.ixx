module;
#include <nlohmann/json.hpp>
export module genesia.dataset.operations;
import std;
export namespace genesia::dataset {
    struct Move final {
        std::filesystem::path source, destination;
        std::string sha, entity;
    };
    void recover_moves(const std::filesystem::path& journal);
    nlohmann::json move_images(std::vector<Move> moves, const std::filesystem::path& journal);
    nlohmann::json undo_moves(const std::filesystem::path& journal);
} // namespace genesia::dataset
