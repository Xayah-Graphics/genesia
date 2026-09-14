module;
#include <nlohmann/json.hpp>
export module genesia.training.samples;
export import genesia.io.safetensors;
export import genesia.data.datasets;
export import genesia.data.images;
export import genesia.training.state;
export import genesia.models.registry;
import genesia.runtime.progress;
import std;
export namespace genesia::training {
    struct Sample final {
        dataset::File file;
        std::vector<std::filesystem::path> paths;
        int label{};
    };
    struct TrainingData final {
        std::string key, fingerprint, issue, training_issue;
        std::filesystem::path root;
        std::vector<std::string> classes;
        std::vector<Sample> samples;
        std::optional<State> training;
        std::optional<models::Descriptor> model;
        std::vector<std::size_t> counts;
    };
    struct Record final {
        std::string id, path, split, group;
        int label{}, width{}, height{};
        std::string rgb_sha;
        std::uint64_t phash{};
        std::vector<std::string> members;
        bool operator==(const Record&) const = default;
    };
    struct Snapshot final {
        std::string fingerprint;
        std::vector<std::string> classes;
        std::vector<Record> records;
        bool operator==(const Snapshot&) const = default;
    };
    void to_json(nlohmann::json& json, const Record& value);
    void from_json(const nlohmann::json& json, Record& value);
    void to_json(nlohmann::json& json, const Snapshot& value);
    void from_json(const nlohmann::json& json, Snapshot& value);
    TrainingData inspect(const dataset::Concept& source, const dataset::Root& root);
    TrainingData inspect(std::string_view key);
    struct Dataset final {
        std::filesystem::path root;
        Snapshot snapshot;
        struct Pixels final {
            std::string cache;
            std::uint64_t offset;
        };
        std::vector<Pixels> pixels;
        std::vector<int> train, val;
        std::map<std::string, std::unique_ptr<files::Mapping>> caches;
        explicit Dataset(const std::filesystem::path& root);
        const unsigned char* rgb(int index) const;
    };
    void prepare(const TrainingData& concept_data, const std::filesystem::path& output, const std::atomic_bool& interrupted, const std::function<void(std::size_t, std::size_t)>& progress);
} // namespace genesia::training
