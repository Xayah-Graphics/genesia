export module genesia.editor.prompt.previews;
import genesia.editor.graphics.renderer;
import genesia.data.images;
import std;

export namespace genesia::editor::previews {
    std::filesystem::path long_path(std::filesystem::path path);

    struct Location final {
        std::filesystem::path root, directory;
        Location(std::filesystem::path root, const std::map<std::string, std::string>& parts);
    };

    struct Images final {
        struct Texture final {
            std::uint64_t id{};
            int width{}, height{};
            bool loading{true}, exists{};
            std::string error;
        };
        struct Undo final {
            std::filesystem::path path, root;
            std::optional<std::vector<std::uint8_t>> bytes;
        };
        std::map<std::filesystem::path, Texture> textures;
        std::optional<Undo> undo;
        std::atomic_bool pending{};
        bool saving{};
        std::string error, status;

        explicit Images(Renderer& renderer);
        ~Images();
        void update(const std::set<std::filesystem::path>& wanted);
        void edit(std::filesystem::path path, std::filesystem::path root, std::optional<std::filesystem::path> source);
        void restore();

    private:
        enum struct Operation { read, replace, clear, restore };
        struct Request final {
            Operation operation{};
            std::filesystem::path path, root, source;
            std::optional<std::vector<std::uint8_t>> bytes;
        };
        struct Result final {
            Operation operation{};
            std::filesystem::path path;
            Image image;
            bool exists{}, applied{};
            std::string error, image_error;
            std::optional<Undo> undo;
        };
        Renderer& renderer;
        std::mutex mutex;
        std::condition_variable condition;
        std::deque<Request> requested;
        std::vector<Result> completed;
        bool closing{};
        std::jthread worker;

        void read();
    };
} // namespace genesia::editor::previews
