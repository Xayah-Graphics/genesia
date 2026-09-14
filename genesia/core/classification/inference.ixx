module;
#include "../models/convnext/kernels.h"
#include <nlohmann/json.hpp>
export module genesia.classification.inference;
export import genesia.models.registry;
export import genesia.models.convnext;
import genesia.data.datasets;
import std;
export namespace genesia::classification {
    struct Result final {
        std::string id, model_sha, image_sha, label;
        std::vector<std::string> classes;
        std::vector<float> scores;
    };
    void to_json(nlohmann::json& json, const Result& result);
    void from_json(const nlohmann::json& json, Result& result);
    struct LoadedClassifier final {
        models::Descriptor descriptor;
        std::unique_ptr<convnext::Network> network;
        std::map<std::pair<int, int>, std::unique_ptr<convnext::NetworkPlan>> plans;
        convnext::GpuResult output{};
        std::size_t offset{};
    };
    struct Pipeline final {
        std::vector<LoadedClassifier> models;
        std::unique_ptr<convnext::NetworkPlan> arena;
        compute::DeviceBuffer result_storage;
        unsigned char* host_results{};
        std::size_t result_bytes{};
        cudaStream_t pending_stream{};
        Pipeline() = default;
        ~Pipeline();
        Pipeline(const Pipeline&) = delete;
        bool prepare(std::vector<models::Descriptor> desired, int width, int height);
        Result run(const models::Descriptor& descriptor, convnext::GpuRgb8 input, cudaStream_t stream, std::string_view image_sha);
    };
    struct Cache final {
        std::map<std::pair<std::string, std::string>, Result> entries;
        std::optional<Result> find(const models::Descriptor& model, std::string_view image_sha);
        void store(const Result& result);
    };
    struct Predictions final {
        Cache cache;
        Pipeline pipeline;
        std::vector<std::string> active;
        Image image;
        std::string image_sha, pixel_sha;
        Result infer(const models::Descriptor& model, const dataset::File& file, bool refresh = false, std::span<const std::uint8_t> rgb = {});
    };
} // namespace genesia::classification
