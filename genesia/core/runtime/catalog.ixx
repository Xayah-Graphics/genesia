export module genesia.runtime.catalog;
export import genesia.data.datasets;
export import genesia.data.captions;
export import genesia.training.samples;
import std;
export namespace genesia::runtime {
    struct CatalogState final {
        std::vector<dataset::Root> roots;
        std::map<std::string, dataset::Concept> concepts;
        std::map<std::string, training::TrainingData> classifiers;
        std::map<std::string, caption::Dataset> captions;
        std::map<std::string, std::string> concept_errors;
        bool ready{};
    };
    struct Catalog final {
        dataset::Index index;
        std::map<std::string, training::TrainingData> classifiers;
        std::set<std::string> described;
        CatalogState load(std::optional<std::string> root = {});
        CatalogState describe(std::string_view key, bool membership = false);
        training::TrainingData& inspect(std::string_view key);
        CatalogState root(std::string_view key);
    };
} // namespace genesia::runtime
