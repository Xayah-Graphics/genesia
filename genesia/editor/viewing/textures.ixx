export module genesia.editor.viewing.textures;
export import genesia.data.images;
import genesia.project;
import genesia.editor.graphics.renderer;
import std;
export namespace genesia::editor {
    struct TextureCache final {
        struct Texture final {
            std::optional<Record> record;
            int width{1}, height{1};
            std::uint64_t texture{}, touched{};
            std::size_t bytes{};
            std::string error, record_error;
            std::filesystem::file_time_type modified;
            std::chrono::steady_clock::time_point checked;
        };
        struct Folder final {
            std::vector<std::filesystem::path> children;
            std::size_t images{};
            bool ready{};
            std::string error;
        };
        std::map<std::filesystem::path, Texture> entries;
        std::map<std::filesystem::path, Folder> folders{{project::output, {}}};
        std::filesystem::path directory{project::output};
        std::vector<std::filesystem::path> history;
        bool history_ready{};
        std::string history_error;
        std::atomic_bool pending{};
        explicit TextureCache(Renderer& renderer);
        ~TextureCache();
        bool receive();
        void adopt(const Record& record, std::uint64_t& texture);
        void refresh(const std::filesystem::path& directory);
        void scan(const std::filesystem::path& directory);
        void request(std::vector<std::filesystem::path> files);
        void discard(const std::filesystem::path& path);
        void shutdown();

    private:
        struct Load final {
            std::filesystem::path path;
            std::optional<std::filesystem::file_time_type> modified;
            bool operator==(const Load&) const = default;
        };
        struct Decoded final {
            std::filesystem::path path;
            Image image;
            std::optional<Record> record;
            std::filesystem::file_time_type modified;
            std::string error, record_error;
            bool unchanged{};
        };
        struct Directory final {
            std::filesystem::path path;
            std::vector<std::filesystem::path> images;
            std::map<std::filesystem::path, Folder> folders;
            std::string error;
        };
        Renderer& renderer;
        std::mutex mutex;
        std::condition_variable condition;
        std::vector<Decoded> results;
        std::vector<Load> requested;
        std::deque<std::filesystem::path> requested_directories{project::output};
        std::vector<Directory> scanned;
        bool closing{};
        std::size_t texture_bytes{};
        std::uint64_t clock{};
        std::jthread reader;
        void read();
    };
} // namespace genesia::editor
