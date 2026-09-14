export module genesia.classification.audit;
export import genesia.classification.inference;
export import genesia.training.samples;
export import genesia.data.transactions;
export import genesia.runtime.progress;
import std;
export namespace genesia::classification {
    struct AuditRow final {
        training::Sample sample;
        std::string label;
        Result prediction;
        float confidence{};
    };
    struct Audit final {
        std::string concept_key, model_sha, fingerprint;
        std::vector<std::string> classes;
        std::vector<AuditRow> rows;
        std::size_t total{};
        bool complete{};
    };
    Audit view(const training::TrainingData& source, Cache& cache, std::string_view category = {});
    Audit audit(training::TrainingData source, Predictions& predictions, bool refresh, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress);
    dataset::MoveResult fix(const training::TrainingData& source, std::string_view sha, std::string_view category);
    dataset::MoveResult undo(const training::TrainingData& source);
} // namespace genesia::classification
