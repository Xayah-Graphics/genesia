module classifier.audit;
import classifier.data.models;
import classifier.training.samples;
import classifier.data.directory;
import tools.files;
import std;
namespace classifier::classification {
    Audit audit(const training::TrainingData& source, const models::Descriptor& model, Predictions& predictions, const bool refresh, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress) {
        if (!source.issue.empty()) throw std::runtime_error{source.issue};
        Audit result{model.sha, source.fingerprint, source.classes};
        for (const auto& sample : source.samples) {
            if (interrupted) throw runtime::Stopped{};
            const auto& label = source.classes[sample.label];
            auto prediction   = predictions.infer(model, sample.file, refresh, progress);
            const auto found  = std::ranges::find(prediction.classes, label);
            if (found == prediction.classes.end()) throw std::runtime_error{"Model has no category: " + label};
            const auto confidence = prediction.scores[found - prediction.classes.begin()];
            result.rows.push_back({sample, label, std::move(prediction), confidence});
            progress({runtime::BatchProgress{runtime::Stage::audit, result.rows.size(), source.samples.size()}});
        }
        std::ranges::sort(result.rows, [](const AuditRow& a, const AuditRow& b) { return std::tie(a.confidence, a.sample.file.path) < std::tie(b.confidence, b.sample.file.path); });
        return result;
    }
    dataset::MoveResult fix(training::TrainingData& source, const std::string_view sha, const std::string_view category, Audit* report) {
        if (!source.issue.empty()) throw std::runtime_error{source.issue};
        if (!std::ranges::contains(source.classes, category)) throw std::runtime_error{"Unknown category: " + std::string(category)};
        const auto sample = std::ranges::find_if(source.samples, [&](const training::Sample& value) { return value.file.sha == sha; });
        if (sample == source.samples.end()) throw std::runtime_error{"Image is no longer in this classifier"};
        if (source.classes[sample->label] == category) throw std::runtime_error{"Image already belongs to this category"};
        AuditRow* row = report ? &*std::ranges::find_if(report->rows, [&](const AuditRow& value) { return value.sample.file.sha == sha; }) : nullptr;
        if (row && !std::ranges::contains(row->prediction.classes, category)) throw std::runtime_error{"Model has no category: " + std::string{category}};
        const auto old_class = source.root / tools::files::path(source.classes[sample->label]);
        std::map<std::filesystem::path, std::size_t> numbers;
        for (const auto& existing : source.samples) {
            if (source.classes[existing.label] != category) continue;
            for (const auto& path : existing.paths) {
                const auto name = tools::files::utf8(path.stem());
                if (name.size() != 5) continue;
                std::size_t number{};
                const auto parsed = std::from_chars(name.data(), name.data() + name.size(), number);
                if (parsed.ec == std::errc{} && parsed.ptr == name.data() + name.size()) numbers[path.parent_path()] = std::max(numbers[path.parent_path()], number);
            }
        }
        std::vector<dataset::Move> moves;
        for (const auto& path : sample->paths) {
            const auto directory = (source.root / tools::files::path(category) / path.lexically_relative(old_class)).parent_path();
            moves.push_back({path, directory / std::format("{:05}.png", ++numbers[directory])});
        }
        auto result = dataset::move_images(std::move(moves));
        --source.counts[sample->label];
        sample->label = int(std::ranges::find(source.classes, category) - source.classes.begin());
        ++source.counts[sample->label];
        sample->paths.clear();
        for (const auto& move : result.paths) sample->paths.push_back(move.destination);
        sample->file.path = sample->paths.front();
        source.fingerprint.clear();
        source.training_issue.clear();
        for (std::size_t i = 0; i < source.counts.size(); ++i)
            if (!source.counts[i]) source.training_issue = "Empty category: " + source.classes[i];
        if (report) {
            row->sample     = *sample;
            row->label      = category;
            row->confidence = row->prediction.scores[std::ranges::find(row->prediction.classes, category) - row->prediction.classes.begin()];
            report->fingerprint.clear();
            std::ranges::sort(report->rows, [](const AuditRow& a, const AuditRow& b) { return std::tie(a.confidence, a.sample.file.path) < std::tie(b.confidence, b.sample.file.path); });
        }
        return result;
    }
    void recycle(training::TrainingData& source, Audit& report, const std::string_view sha, const std::function<void(const std::filesystem::path&)>& file, const std::atomic_bool& interrupted) {
        const auto sample = std::ranges::find_if(source.samples, [&](const training::Sample& value) { return value.file.sha == sha; });
        const auto row    = std::ranges::find_if(report.rows, [&](const AuditRow& value) { return value.sample.file.sha == sha; });
        const auto count  = sample->paths.size();
        std::exception_ptr failure;
        try {
            while (!sample->paths.empty()) {
                if (interrupted) throw runtime::Stopped{};
                file(sample->paths.back());
                sample->paths.pop_back();
            }
        } catch (...) {
            failure = std::current_exception();
        }
        if (sample->paths.size() != count) {
            source.fingerprint.clear();
            report.fingerprint.clear();
            if (sample->paths.empty()) {
                if (!--source.counts[sample->label]) source.training_issue = "Empty category: " + source.classes[sample->label];
                source.samples.erase(sample);
                report.rows.erase(row);
            } else {
                sample->file.path = sample->paths.front();
                row->sample       = *sample;
            }
        }
        if (failure) std::rethrow_exception(failure);
    }
} // namespace classifier::classification
