export module genesia.data.transactions;
export import genesia.data.datasets;
import std;
export namespace genesia::dataset {
    struct MoveResult final {
        std::size_t moved{}, restored{};
        std::vector<Move> paths;
    };
    MoveResult recover_moves(const std::filesystem::path& journal);
    MoveResult move_images(std::vector<Move> moves, const std::filesystem::path& journal);
    MoveResult undo_moves(const std::filesystem::path& journal);
} // namespace genesia::dataset
