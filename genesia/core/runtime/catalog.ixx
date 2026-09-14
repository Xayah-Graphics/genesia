export module genesia.runtime.catalog;
export import genesia.data.datasets;
export import genesia.training.samples;
import std;
export namespace genesia::runtime {
    struct CatalogState final {
        std::vector<dataset::Root> roots;
        std::map<std::string, dataset::Concept> concepts;
        std::map<std::string, training::TrainingData> classifiers;
        std::map<std::string, std::string> concept_errors;
        std::uint64_t revision{};
        bool ready{};
        std::string error;
    };
    struct Catalog final {
        std::atomic_bool pending{};
        explicit Catalog(std::function<void()> notify = {});
        ~Catalog();
        std::shared_ptr<const CatalogState> poll();
        void refresh();

    private:
        std::function<void()> notify;
        std::mutex mutex;
        std::shared_ptr<const CatalogState> update;
        std::atomic_bool rescan{true};
        std::jthread worker;
        void scan(std::stop_token stop);
    };
} // namespace genesia::runtime
