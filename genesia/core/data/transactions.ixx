export module genesia.data.transactions;
import std;
export namespace genesia::dataset {
    struct Move final {
        std::filesystem::path source, destination;
        std::string sha, entity;
    };
    struct MoveResult final {
        std::size_t moved{}, restored{};
    };
    void recover_moves(const std::filesystem::path& journal);
    MoveResult move_images(std::vector<Move> moves, const std::filesystem::path& journal);
    MoveResult undo_moves(const std::filesystem::path& journal);
} // namespace genesia::dataset
