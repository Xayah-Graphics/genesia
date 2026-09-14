module;
#include <Windows.h>
export module genesia.editor.library;
export import genesia.dataset;
export import genesia.classifier.dataset;
import genesia.generation.output;
import genesia.images;
import genesia.editor.ui.renderer;
import std;

export namespace genesia::editor {
    struct Library final {
        struct Texture final {
            dataset::File file;
            Record record;
            std::uint64_t texture{}, touched{};
            std::size_t bytes{};
            std::string error;
        };
        std::vector<dataset::Root> roots;
        std::map<std::string, dataset::Concept> concepts;
        std::map<std::string, classifier::TrainingData> classifiers;
        std::map<std::string, std::string> concept_errors;
        std::map<std::string, Texture> textures;
        std::uint64_t revision{};
        bool ready{};
        std::string error;
        std::atomic_bool pending{};

        Library(std::shared_ptr<const prompt::Catalog> catalog, Renderer& renderer);
        ~Library();
        void receive();
        void request(std::vector<dataset::File> files);
        void refresh();

    private:
        struct Decoded final {
            dataset::File file;
            Image image;
            Record record;
            std::string error;
        };
        const std::shared_ptr<const prompt::Catalog> catalog;
        Renderer& renderer;
        std::unique_ptr<std::remove_pointer_t<HANDLE>, decltype(&CloseHandle)> wake{nullptr, CloseHandle};
        std::mutex mutex;
        std::optional<std::vector<dataset::Root>> index;
        std::map<std::string, dataset::Concept> concept_updates;
        std::map<std::string, classifier::TrainingData> classifier_updates;
        std::map<std::string, std::string> concept_error_updates;
        std::vector<Decoded> results;
        std::vector<dataset::File> requested;
        bool rescan{true};
        std::string failure;
        std::size_t texture_bytes{};
        std::uint64_t clock{};
        std::jthread worker;
        void read(std::stop_token stop);
    };
} // namespace genesia::editor
