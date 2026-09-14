module genesia.classification.audit;
import genesia.models.registry;
import genesia.training.samples;
import genesia.data.transactions;
import genesia.io.files;
import genesia.io.hash;
import std;
namespace genesia::classification {
    Audit view(const training::TrainingData& source, Cache& cache, const std::string_view category) {
        Audit result{source.key, source.model ? source.model->sha : std::string{}, source.fingerprint, source.classes};
        if (!source.model) return result;
        if (!category.empty() && !std::ranges::contains(source.classes, category)) throw std::runtime_error{"Unknown audit category: " + std::string{category}};
        for (const auto& sample : source.samples) {
            const auto& label = source.classes[sample.label];
            if (!category.empty() && category != label) continue;
            ++result.total;
            auto prediction = cache.find(*source.model, sample.file.sha);
            if (!prediction) continue;
            const auto found = std::ranges::find(prediction->classes, label);
            if (found == prediction->classes.end()) throw std::runtime_error{"Model has no category: " + label};
            const auto confidence = prediction->scores[found - prediction->classes.begin()];
            result.rows.push_back({sample, label, std::move(*prediction), confidence});
        }
        std::ranges::sort(result.rows, [](const AuditRow& a, const AuditRow& b) { return std::tie(a.confidence, a.sample.file.path) < std::tie(b.confidence, b.sample.file.path); });
        result.complete = result.rows.size() == result.total;
        return result;
    }
    Audit audit(training::TrainingData source, Predictions& predictions, const bool refresh, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress) {
        if (!source.issue.empty()) throw std::runtime_error{source.issue};
        if (!source.model) throw std::runtime_error{"Concept has no published classifier: " + source.key};
        const auto descriptor = *source.model;
        std::size_t completed{};
        for (const auto& sample : source.samples) {
            if (interrupted.load()) throw runtime::Stopped{};
            predictions.infer(descriptor, sample.file, refresh);
            progress({runtime::BatchProgress{runtime::Stage::audit, ++completed, source.samples.size()}});
        }
        source.model = descriptor;
        return view(source, predictions.cache);
    }
    dataset::MoveResult fix(const training::TrainingData& source, const std::string_view sha, const std::string_view category) {
        if (!source.issue.empty()) throw std::runtime_error{source.issue};
        if (!std::ranges::contains(source.classes, category)) throw std::runtime_error{"Unknown category: " + std::string(category)};
        const auto sample = std::ranges::find_if(source.samples, [&](const training::Sample& value) { return value.file.sha == sha; });
        if (sample == source.samples.end()) throw std::runtime_error{"Image is no longer in this concept"};
        if (source.classes[sample->label] == category) throw std::runtime_error{"Image already belongs to this category"};
        const auto old_class = source.root / files::path(source.classes[sample->label]);
        std::vector<dataset::Move> moves;
        for (const auto& path : sample->paths) moves.push_back({path, source.root / files::path(category) / path.lexically_relative(old_class), sample->file.sha});
        return dataset::move_images(std::move(moves), source.root / ".genesia" / "audit-moves.json");
    }
    dataset::MoveResult undo(const training::TrainingData& source) {
        return dataset::undo_moves(source.root / ".genesia" / "audit-moves.json");
    }
} // namespace genesia::classification
