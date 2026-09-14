module;
#include <nlohmann/json.hpp>
export module genesia.classifier.audit;
export import genesia.classifier.inference;
import std;
export namespace genesia::classifier {
    nlohmann::json audit(std::string_view key, Predictions& predictions, bool refresh, const std::atomic_bool& interrupted, const std::function<void(const nlohmann::json&)>& progress, const std::function<void()>& yield);
    nlohmann::json fix(std::string_view key, std::string_view sha, std::string_view category);
    nlohmann::json undo(std::string_view key);
} // namespace genesia::classifier
