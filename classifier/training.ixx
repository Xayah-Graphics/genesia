module;
#include <nlohmann/json.hpp>
export module classifier.training;
export import classifier.network;
export import classifier.dataset;
import std;
export namespace classifier {
    struct TrainingConfig {
        int physical_batch = 4, effective_batch = 64, head_only_steps = 25, warmup_steps = 10;
        float head_only_lr = .001f, backbone_lr = .00003f, head_lr = .0003f, weight_decay = .01f, clip_norm = 1;
        std::uint64_t seed = 42;
        int eval_interval = 25, save_interval = 25, log_interval = 5;
    };
    struct TrainingOptions {
        std::filesystem::path dataset;
        int steps                = 400;
        bool restart             = false;
        nlohmann::json overrides = nlohmann::json::object();
    };
    nlohmann::json evaluate(Network& model, Dataset& dataset, const std::filesystem::path& output, int step);
    void train(const TrainingOptions& options, const std::atomic_bool& interrupted);
} // namespace classifier
