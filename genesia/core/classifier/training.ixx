module;
#include <nlohmann/json.hpp>
export module genesia.classifier.training;
export import genesia.classifier.network;
export import genesia.classifier.dataset;
import std;
export namespace genesia::classifier {
    struct TrainingConfig {
        int physical_batch = 4, effective_batch = 64, head_only_steps = 25, warmup_steps = 10;
        float head_only_lr = .001f, backbone_lr = .00003f, head_lr = .0003f, weight_decay = .01f, clip_norm = 1;
        std::uint64_t seed = 42;
        int eval_interval = 25, save_interval = 25, log_interval = 5;
    };
    struct TrainingOptions {
        std::string concept_key;
        int steps                = 400;
        bool restart             = false;
        nlohmann::json overrides = nlohmann::json::object();
    };
    nlohmann::json evaluate(Network& model, Dataset& dataset, const std::filesystem::path& output, int step);
    nlohmann::json train(const TrainingOptions& options, const std::atomic_bool& interrupted, const std::function<void(const nlohmann::json&)>& progress, const std::function<void()>& yield);
    nlohmann::json configuration();
} // namespace genesia::classifier
