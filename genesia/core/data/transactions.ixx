export module genesia.data.transactions;
export import genesia.data.datasets;
import std;
export namespace genesia::dataset {
    struct MoveResult final {
        std::size_t moved{}, restored{};
        std::vector<Move> paths;
    };
    struct DeleteResult final {
        std::string root, sha;
        std::vector<std::filesystem::path> paths;
        std::string error;
    };
    struct NormalizeResult final {
        std::string root;
        std::size_t files{}, linked{};
        std::vector<Move> renamed;
        bool stopped{};
        std::string error;
    };
    enum class NormalizeStage { linking, renaming };
    MoveResult recover_moves(const std::filesystem::path& journal);
    MoveResult move_images(std::vector<Move> moves, const std::filesystem::path& journal);
    MoveResult undo_moves(const std::filesystem::path& journal);
    DeleteResult delete_image(Index& index, std::string_view root, std::string_view sha);
    void recover_normalization(const std::filesystem::path& root);
    NormalizeResult normalize(Index& index, std::string_view root, const std::atomic_bool& interrupted, const std::function<void(NormalizeStage, std::size_t, std::size_t)>& progress);
} // namespace genesia::dataset
