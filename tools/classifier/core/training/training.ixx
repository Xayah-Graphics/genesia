export module classifier.training;
export import classifier.training.samples;
export import classifier.training.state;
export import classifier.progress;
import std;
export namespace classifier::training {
    struct Options final {
        std::filesystem::path root;
        int steps{400};
        bool restart{};
        std::optional<Config> config;
    };
    void train(const Options& options, const TrainingData& source, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress);
} // namespace classifier::training
