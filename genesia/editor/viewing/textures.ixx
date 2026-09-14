export module genesia.editor.viewing.textures;
export import genesia.data.datasets;
import genesia.data.images;
import genesia.editor.graphics.renderer;
import std;
export namespace genesia::editor {
    struct TextureCache final {
        struct Texture final {
            dataset::File file;
            Record record;
            std::uint64_t texture{}, touched{};
            std::size_t bytes{};
            std::string error;
        };
        std::map<std::string, Texture> entries;
        std::atomic_bool pending{};
        explicit TextureCache(Renderer& renderer);
        ~TextureCache();
        void receive();
        void adopt(const dataset::File& file, const Record& record, std::uint64_t& texture);
        void request(std::vector<dataset::File> files);
        void discard(std::span<const std::filesystem::path> paths);

    private:
        struct Decoded final {
            dataset::File file;
            Image image;
            Record record;
            std::string error;
        };
        Renderer& renderer;
        std::mutex mutex;
        std::condition_variable condition;
        std::vector<Decoded> results;
        std::vector<dataset::File> requested;
        bool closing{};
        std::size_t texture_bytes{};
        std::uint64_t clock{};
        std::jthread worker;
        void read();
    };
} // namespace genesia::editor
