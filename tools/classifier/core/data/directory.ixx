export module classifier.data.directory;
export import tools.images;
export import tools.files;
import std;
export namespace classifier::dataset {
    struct File final {
        std::filesystem::path path;
        std::string sha;
        int width{}, height{};
    };
    File read_file(const std::filesystem::path& path);
    struct Index final {
        explicit Index(const std::filesystem::path& root);
        File identify(const std::filesystem::path& path);
        void flush();

    private:
        struct Cached final {
            std::int64_t modified{};
            std::uint64_t bytes{};
            std::string sha;
            int width{}, height{};
        };
        std::map<std::string, Cached> entries;
        std::set<std::string> visited;
        std::filesystem::path cache;
        bool dirty{};
    };
    struct Move final {
        std::filesystem::path source, destination;
    };
    struct MoveResult final {
        std::vector<Move> paths;
    };
    MoveResult move_images(std::vector<Move> moves);
} // namespace classifier::dataset
