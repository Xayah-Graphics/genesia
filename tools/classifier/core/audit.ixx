export module classifier.audit;
export import classifier.inference;
export import classifier.training.samples;
export import classifier.progress;
import std;
export namespace classifier::classification {
    struct AuditRow final {
        training::Sample sample;
        std::string label;
        Result prediction;
        float confidence{};
    };
    struct Audit final {
        std::string model_sha, fingerprint;
        std::vector<std::string> classes;
        std::vector<AuditRow> rows;
    };
    Audit audit(const training::TrainingData& source, const models::Descriptor& model, Predictions& predictions, bool refresh, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress);
    dataset::MoveResult fix(training::TrainingData& source, std::string_view sha, std::string_view category, Audit* report = nullptr);
    void recycle(training::TrainingData& source, Audit& report, std::string_view sha, const std::function<void(const std::filesystem::path&)>& file, const std::atomic_bool& interrupted);
} // namespace classifier::classification
