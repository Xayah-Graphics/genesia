module;
#include <nlohmann/json.hpp>
export module genesia.classifier.classify;
export import genesia.classifier.inference;
import std;
export namespace genesia::classifier {
    nlohmann::json classify(std::string_view key, const std::filesystem::path& input, Predictions& predictions, const std::atomic_bool& interrupted, const std::function<void(const nlohmann::json&)>& progress, const std::function<void()>& yield);
}
