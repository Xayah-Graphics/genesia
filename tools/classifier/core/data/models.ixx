export module classifier.data.models;
import std;
export namespace classifier::models {
    struct Descriptor final {
        std::filesystem::path path;
        std::string sha;
        bool operator==(const Descriptor&) const = default;
    };
    struct Info final {
        std::filesystem::path root;
        std::optional<Descriptor> model;
        std::vector<std::string> classes;
        int step{};
    };
    std::optional<Descriptor> find(const std::filesystem::path& root);
    Descriptor resolve(const std::filesystem::path& root);
    Info inspect(const std::filesystem::path& root);
    void publish(const std::filesystem::path& root, const std::filesystem::path& file, std::string fingerprint, int step);
    void unpublish(const std::filesystem::path& root);
} // namespace classifier::models
