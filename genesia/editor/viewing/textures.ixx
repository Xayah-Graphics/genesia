export module genesia.editor.viewing.textures;
export import genesia.data.images;
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
        std::map<std::filesystem::path, Texture> entries;
        std::vector<std::filesystem::path> history;
        bool history_ready{};
        std::string history_error;
        std::atomic_bool pending{};
        explicit TextureCache(Renderer& renderer);
        ~TextureCache();
        bool receive();
        void adopt(const Record& record, std::uint64_t& texture);
        void refresh();
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
        Renderer& renderer;
        std::mutex mutex;
        std::condition_variable condition;
        std::vector<Decoded> results;
        std::vector<Load> requested;
        std::vector<std::filesystem::path> scanned;
        bool closing{}, scanned_ready{};
        bool rescan{true};
        std::string scan_error;
        std::size_t texture_bytes{};
        std::uint64_t clock{};
        std::jthread reader;
        void read();
    };
} // namespace genesia::editor
