module;
#include "kernels.h"
#include <nlohmann/json.hpp>
export module genesia.classifier.inference;
export import genesia.classifier.network;
import genesia.dataset;
import std;
export namespace genesia::classifier {
    struct Descriptor final {
        std::string id, sha;
        std::filesystem::path path;
        bool operator==(const Descriptor&) const = default;
    };
    struct Result final {
        std::string id, model_sha, image_sha, label;
        std::vector<std::string> classes;
        std::vector<float> scores;
    };
    void to_json(nlohmann::json& json, const Result& result);
    void from_json(const nlohmann::json& json, Result& result);
    Descriptor model(std::string_view key);
    struct LoadedClassifier final {
        Descriptor descriptor;
        std::unique_ptr<Network> network;
        std::map<std::pair<int, int>, std::unique_ptr<NetworkPlan>> plans;
        GpuResult output{};
        std::size_t offset{};
    };
    struct Pipeline final {
        std::vector<LoadedClassifier> models;
        std::unique_ptr<NetworkPlan> arena;
        DeviceBuffer result_storage;
        unsigned char* host_results{};
        std::size_t result_bytes{};
        cudaStream_t pending_stream{};
        Pipeline() = default;
        ~Pipeline();
        Pipeline(const Pipeline&) = delete;
        void prepare(std::vector<Descriptor> desired, int width, int height);
        Result run(const Descriptor& descriptor, GpuRgb8 input, cudaStream_t stream, std::string_view image_sha);
    };
    struct Predictions final {
        std::map<std::string, nlohmann::json> cached;
        Pipeline pipeline;
        std::optional<Result> find(const Descriptor& model, std::string_view image_sha);
        Result infer(const Descriptor& model, const dataset::File& file, bool refresh = false);
    };
} // namespace genesia::classifier
