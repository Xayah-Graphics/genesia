module;
#include <Windows.h>
export module genesia.editor.gallery;
import genesia.generation.output;
import genesia.prompt;
import genesia.editor.runtime.images;
import std;

export namespace genesia::editor {
    struct Gallery final {
        struct File final {
            std::uint64_t id{}, modified{}, bytes{};
            std::filesystem::path path;
            bool operator==(const File&) const = default;
        };
        struct Result final {
            File file;
            std::shared_ptr<const Image> image;
            std::optional<Record> record;
            std::uint64_t ticket{};
            std::string error;
        };

        std::vector<File> files;
        std::unordered_map<std::uint64_t, std::size_t> positions;
        std::uint64_t revision{};
        bool ready{};
        std::string error;
        std::atomic_bool pending{};

        Gallery(std::filesystem::path directory, std::shared_ptr<const prompt::Catalog> catalog);
        ~Gallery();
        std::vector<Result> receive();
        std::uint64_t select(const File& file);
        void cancel();
        void thumbnails(std::vector<File> files);
        void refresh();

    private:
        const std::filesystem::path directory;
        const std::shared_ptr<const prompt::Catalog> catalog;
        std::unique_ptr<std::remove_pointer_t<HANDLE>, decltype(&CloseHandle)> wake{nullptr, CloseHandle};
        std::mutex mutex;
        std::optional<std::vector<File>> index;
        std::vector<Result> results;
        std::vector<File> requested_thumbnails;
        std::optional<File> requested_image;
        std::uint64_t ticket{};
        bool rescan{true};
        std::string failure;
        std::jthread worker;

        void read(std::stop_token stop);
    };
} // namespace genesia::editor
