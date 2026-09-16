export module genesia.editor.prompt.previews;
import genesia.editor.graphics.renderer;
import genesia.data.images;
import std;

export namespace genesia::editor::previews {
    std::filesystem::path long_path(std::filesystem::path path);

    struct Location final {
        std::filesystem::path root, directory;
        Location(std::filesystem::path root, const std::map<std::string, std::string>& parts);
        std::filesystem::path scene(std::string_view name, const std::map<std::string, std::string>& variations) const;
    };

    struct Images final {
        struct Texture final {
            std::uint64_t id{};
            int width{}, height{};
            bool loading{true}, exists{};
            std::string error;
        };
        std::map<std::filesystem::path, Texture> textures;
        std::atomic_bool pending{};
        bool saving{};
        std::string error, status;

        explicit Images(Renderer& renderer);
        ~Images();
        void update(const std::set<std::filesystem::path>& wanted);
        const Texture& request(const std::filesystem::path& path);
        void edit(std::filesystem::path path, std::filesystem::path root, std::optional<std::filesystem::path> source);

    private:
        enum struct Operation { read, replace, clear };
        struct Request final {
            Operation operation{};
            std::filesystem::path path, root, source;
        };
        struct Result final {
            Operation operation{};
            std::filesystem::path path;
            Image image;
            bool exists{}, applied{};
            std::string error, image_error;
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
