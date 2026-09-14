module;
#include <nlohmann/json.hpp>
module genesia.classifier.audit;
import genesia.classifier.dataset;
import genesia.dataset.operations;
import genesia.files;
import genesia.hash;
import std;
namespace genesia::classifier {
    nlohmann::json audit(const std::string_view key, Predictions& predictions, const bool refresh, const std::atomic_bool& interrupted, const std::function<void(const nlohmann::json&)>& progress, const std::function<void()>& yield) {
        const dataset::Lock lock{"concept-" + sha256({reinterpret_cast<const unsigned char*>(key.data()), key.size()})};
        auto source = inspect(key);
        {
            const dataset::Lock recovery{"image-moves"};
            dataset::recover_moves(source.root / ".genesia" / "classifier" / "audit-moves.json");
        }
        source = inspect(key);
        if (!source.issue.empty()) throw std::runtime_error{source.issue};
        const auto descriptor = model(key);
        nlohmann::json report{{"concept", key}, {"model_sha", descriptor.sha}, {"fingerprint", source.fingerprint}, {"classes", source.classes}};
        std::vector<nlohmann::json> rows;
        for (const auto& sample : source.samples) {
            if (interrupted.load()) throw std::runtime_error{"Audit stopped"};
            const auto result = predictions.infer(descriptor, sample.file, refresh);
            const auto& label = source.classes[sample.label];
            const auto found  = std::ranges::find(result.classes, label);
            if (found == result.classes.end()) throw std::runtime_error{"Model has no category: " + label};
            const auto confidence = result.scores[found - result.classes.begin()];
            std::vector<std::string> paths;
            for (const auto& path : sample.paths) paths.push_back(files::utf8(path));
            rows.push_back({{"paths", paths}, {"sha", sample.file.sha}, {"path", files::utf8(sample.file.path)}, {"label", label}, {"predicted", result.label}, {"confidence", confidence}, {"classes", result.classes}, {"scores", result.scores}});
            progress({{"stage", "audit"}, {"completed", rows.size()}, {"total", source.samples.size()}});
            yield();
        }
        std::ranges::sort(rows, [](const auto& a, const auto& b) { return a.at("confidence") == b.at("confidence") ? a.at("path") < b.at("path") : a.at("confidence") < b.at("confidence"); });
        report["rows"]     = rows;
        report["complete"] = true;
        files::write_json(source.root / ".genesia" / "classifier" / "audit.json", report);
        return report;
    }
    nlohmann::json fix(const std::string_view key, const std::string_view sha, const std::string_view category) {
        const dataset::Lock lock{"concept-" + sha256({reinterpret_cast<const unsigned char*>(key.data()), key.size()})};
        auto source = inspect(key);
        {
            const dataset::Lock recovery{"image-moves"};
            dataset::recover_moves(source.root / ".genesia" / "classifier" / "audit-moves.json");
        }
        source = inspect(key);
        if (!source.issue.empty()) throw std::runtime_error{source.issue};
        if (!std::ranges::contains(source.classes, category)) throw std::runtime_error{"Unknown category: " + std::string(category)};
        const auto sample = std::ranges::find_if(source.samples, [&](const Sample& value) { return value.file.sha == sha; });
        if (sample == source.samples.end()) throw std::runtime_error{"Image is no longer in this concept"};
        if (source.classes[sample->label] == category) throw std::runtime_error{"Image already belongs to this category"};
        const auto old_class = source.root / files::path(source.classes[sample->label]);
        std::vector<dataset::Move> moves;
        for (const auto& path : sample->paths) moves.push_back({path, source.root / files::path(category) / path.lexically_relative(old_class), sample->file.sha, sample->file.entity});
        return dataset::move_images(std::move(moves), source.root / ".genesia" / "classifier" / "audit-moves.json");
    }
    nlohmann::json undo(const std::string_view key) {
        const dataset::Lock lock{"concept-" + sha256({reinterpret_cast<const unsigned char*>(key.data()), key.size()})};
        auto source = inspect(key);
        {
            const dataset::Lock recovery{"image-moves"};
            dataset::recover_moves(source.root / ".genesia" / "classifier" / "audit-moves.json");
        }
        source = inspect(key);
        return dataset::undo_moves(source.root / ".genesia" / "classifier" / "audit-moves.json");
    }
} // namespace genesia::classifier
