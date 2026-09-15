export module genesia.editor.viewing.textures;
export import genesia.data.datasets;
import genesia.segmentation.foreground;
import genesia.data.images;
import genesia.editor.graphics.renderer;
import std;
export namespace genesia::editor {
    struct TextureCache final {
        struct Texture final {
            dataset::File file;
            std::optional<Record> record;
            std::uint64_t texture{}, touched{};
            std::size_t bytes{};
            std::string error;
            std::string record_error;
            std::uint64_t mask{};
            bool mask_checked{};
            std::string mask_error;
        };
        std::map<std::string, Texture> entries;
        std::atomic_bool pending{};
        bool paused{};
        explicit TextureCache(Renderer& renderer);
        ~TextureCache();
        void receive();
        void adopt(const dataset::File& file, const Record& record, std::uint64_t& texture);
        void request(std::vector<dataset::File> files, bool masks = false);
        void mask_ready(const std::string& sha);
        void discard(std::span<const std::filesystem::path> paths);
        void relocate(std::span<const dataset::Move> moves);

    private:
        struct Load final {
            dataset::File file;
            bool mask{};
            bool operator==(const Load&) const = default;
        };
        struct Decoded final {
            dataset::File file;
            Image image;
            std::optional<Record> record;
            std::string error;
            std::string record_error;
            bool mask{};
        };
        Renderer& renderer;
        std::mutex mutex;
        std::condition_variable condition;
        std::vector<Decoded> results;
        std::vector<Load> requested;
        bool closing{};
        std::size_t texture_bytes{};
        std::uint64_t clock{};
        std::jthread worker;
        void read();
    };
} // namespace genesia::editor
