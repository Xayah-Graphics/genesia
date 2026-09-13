module;
export module classifier.classify;
import std;
export namespace classifier {
    struct ClassifyOptions {
        std::filesystem::path dataset;
        std::filesystem::path input;
        std::filesystem::path model;
    };
    void classify(const ClassifyOptions& options, const std::atomic_bool& interrupted);
} // namespace classifier