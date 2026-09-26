export module classifier.classify;
export import classifier.inference;
export import classifier.progress;
import std;
export namespace classifier::classification {
    struct Classification final {
        std::filesystem::path input;
        dataset::MoveResult movement;
        std::map<std::string, std::size_t> classes;
    };
    Classification classify(const models::Descriptor& model, const std::filesystem::path& input, Predictions& predictions, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress);
} // namespace classifier::classification
