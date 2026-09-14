module;
#include <nlohmann/json.hpp>
export module genesia.classifier.dataset;
export import genesia.classifier.storage;
export import genesia.dataset;
export import genesia.images;
import std;
export namespace genesia::classifier {
    struct Sample final {
        dataset::File file;
        std::vector<std::filesystem::path> paths;
        int label{};
    };
    struct TrainingData final {
        std::string key, fingerprint, issue;
        std::filesystem::path root;
        std::vector<std::string> classes;
        std::vector<Sample> samples;
        nlohmann::json training;
    };
    nlohmann::json read_training(const dataset::Concept& source);
    TrainingData inspect(const dataset::Concept& source, const dataset::Root& root);
    TrainingData inspect(std::string_view key);
    struct Record final {
        std::string id, path, split, group, cache;
        int label, width, height;
        std::uint64_t offset;
    };
    struct Dataset final {
        std::filesystem::path root;
        nlohmann::json metadata;
        std::vector<std::string> classes;
        std::vector<Record> records;
        std::vector<int> train, val;
        std::map<std::string, std::unique_ptr<Mapping>> caches;
        explicit Dataset(const std::filesystem::path& root);
        const unsigned char* rgb(int index) const;
    };
    void prepare(const TrainingData& concept_data, const std::filesystem::path& output, const std::atomic_bool& interrupted, const std::function<void(std::size_t, std::size_t)>& progress);
} // namespace genesia::classifier
