export module classifier.audit;
import std;
export namespace classifier {
    void audit(const std::filesystem::path& dataset, const std::filesystem::path& model, const std::atomic_bool& interrupted);
}
