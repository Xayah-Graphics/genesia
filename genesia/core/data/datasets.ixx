module;
#include <nlohmann/json.hpp>
export module genesia.data.datasets;
export import genesia.project;
export import genesia.data.concepts;
export import genesia.data.images;
export import genesia.io.files;
import std;

export namespace genesia::dataset {
    struct File final {
        std::filesystem::path path;
        std::string entity, sha;
        std::int64_t modified{};
        std::uint64_t bytes{};
        int width{}, height{};
        bool operator==(const File&) const = default;
    };
    struct Collection final {
        std::string key, name;
        std::vector<File> images;
    };
    struct Root final {
        Collection all;
        std::vector<Collection> concepts;
        std::vector<File> files;
        std::vector<std::vector<std::filesystem::path>> conflicts;
        std::string error;
        bool ready{};
    };
    struct Move final {
        std::filesystem::path source, destination;
        std::string sha;
    };
    struct Index final {
        std::vector<Root> roots;
        std::uint64_t next_output{1};
        void flush();
        void scan(std::string_view name);
        File identify(const std::filesystem::path& path, const Record* record = nullptr);
        void insert(File file);
        std::vector<std::string> apply(std::span<const Move> moves);
        void rebuild(Root& root);

    private:
        struct Cached final {
            std::int64_t modified{};
            std::uint64_t bytes{};
            std::string sha;
            int width{}, height{};
        };
        std::map<std::string, Cached> cache;
        bool loaded{}, dirty{};
        void load_cache();
    };
} // namespace genesia::dataset
