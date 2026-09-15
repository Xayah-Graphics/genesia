export module genesia.data.operations;
export import genesia.data.datasets;
import std;
export namespace genesia::dataset {
    struct MoveResult final {
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
    MoveResult move_images(std::vector<Move> moves);
    DeleteResult delete_image(Index& index, std::string_view root, std::string_view sha);
    NormalizeResult normalize(Index& index, std::string_view root, const std::atomic_bool& interrupted, const std::function<void(NormalizeStage, std::size_t, std::size_t)>& progress);
} // namespace genesia::dataset
