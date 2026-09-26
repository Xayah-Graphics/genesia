module;
#include "models/convnext/kernels.h"
#include <nlohmann/json.hpp>
export module classifier.inference;
export import classifier.data.models;
export import classifier.data.directory;
export import classifier.models.convnext;
export import classifier.progress;
import std;
export namespace classifier::classification {
    struct Result final {
        std::string model_sha, image_sha, label;
        std::vector<std::string> classes;
        std::vector<float> scores;
    };
    void to_json(nlohmann::json& json, const Result& value);
    void from_json(const nlohmann::json& json, Result& value);
    struct Cache final {
        std::map<std::pair<std::string, std::string>, Result> entries;
        std::optional<Result> find(const models::Descriptor& model, std::string_view image_sha);
        void store(const Result& result);
    };
    struct Predictions final {
        Cache cache;
        std::optional<models::Descriptor> descriptor;
        std::unique_ptr<convnext::Network> network;
        std::unique_ptr<convnext::NetworkPlan> arena;
        std::map<std::pair<int, int>, std::unique_ptr<convnext::NetworkPlan>> plans;
        tools::compute::DeviceBuffer output;
        Result infer(const models::Descriptor& model, const dataset::File& file, bool refresh, const std::function<void(const runtime::Progress&)>& progress);
    };
} // namespace classifier::classification
