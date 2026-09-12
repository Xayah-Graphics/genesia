module;
#include "kernels.h"

#include <nlohmann/json.hpp>
export module classifier.inference;
export import classifier.network;
import std;
export namespace classifier {
    struct Descriptor {
        std::string id;
        std::filesystem::path path;
        std::filesystem::file_time_type modified;
        bool operator==(const Descriptor&) const = default;
    };
    struct Selection {
        std::vector<std::string> enabled;
        bool discard_failed = false;
    };
    struct Result {
        std::string id, label;
        std::vector<std::string> classes;
        std::vector<float> scores;
        float threshold = .9f;
        bool accepted   = false;
        std::string error;
    };
    struct Results {
        std::vector<Result> classifiers;
        bool passed = true;
        std::string error;
    };
    std::filesystem::path model_directory();
    void to_json(nlohmann::json& json, const Result& result);
    void from_json(const nlohmann::json& json, Result& result);
    void to_json(nlohmann::json& json, const Results& results);
    void from_json(const nlohmann::json& json, Results& results);
    std::vector<Descriptor> discover();
    struct LoadedClassifier {
        Descriptor descriptor;
        std::unique_ptr<Network> network;
        std::map<std::pair<int, int>, std::unique_ptr<NetworkPlan>> plans;
        GpuResult output{};
        std::size_t offset = 0;
        float threshold    = .9f;
    };
    struct Pipeline {
        std::vector<LoadedClassifier> models;
        // This owner is retained until every plan referencing its buffers is destroyed.
        std::unique_ptr<NetworkPlan> arena;
        DeviceBuffer result_storage;
        unsigned char* host_results = nullptr;
        std::size_t result_bytes    = 0;
        cudaStream_t pending_stream = nullptr;
        Pipeline()                  = default;
        ~Pipeline();
        Pipeline(const Pipeline&) = delete;
        void prepare(const std::vector<Descriptor>& catalog, const Selection& selection, int width, int height);
        Results run(GpuRgb8 input, cudaStream_t stream);
    };
} // namespace classifier
