export module genesia.classification.classify;
export import genesia.classification.inference;
export import genesia.data.operations;
export import genesia.runtime.progress;
import std;
export namespace genesia::classification {
    struct Classification final {
        std::filesystem::path input;
        dataset::MoveResult movement;
        std::map<std::string, std::size_t> classes;
    };
    Classification classify(dataset::Index& index, std::string_view key, const std::filesystem::path& input, Predictions& predictions, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress);
} // namespace genesia::classification
