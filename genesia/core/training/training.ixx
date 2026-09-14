export module genesia.training;
export import genesia.training.samples;
export import genesia.training.state;
export import genesia.runtime.progress;
import std;
export namespace genesia::training {
    struct Options final {
        std::string concept_key;
        int steps{400};
        bool restart{};
        std::optional<Config> config;
    };
    State train(const Options& options, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress, const std::function<void()>& yield);
} // namespace genesia::training
