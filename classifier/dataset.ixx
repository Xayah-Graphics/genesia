module;
#include <nlohmann/json.hpp>
export module classifier.dataset;
export import classifier.storage;
import std;
export namespace classifier {
    struct Record {
        std::string id, path, split, group, cache;
        int label, width, height;
        std::uint64_t offset;
    };
    struct Dataset {
        std::filesystem::path root;
        nlohmann::json metadata;
        std::vector<std::string> classes;
        std::vector<Record> records;
        std::vector<int> train, val;
        std::map<std::string, std::unique_ptr<Mapping>> caches;
        explicit Dataset(const std::filesystem::path& root);
        const unsigned char* rgb(int index) const;
    };
    struct Image {
        int width, height;
        std::vector<unsigned char> rgb;
    };
    Image load_image(const std::filesystem::path& path);
    void prepare(const std::vector<std::filesystem::path>& folders, const std::filesystem::path& output);
    std::vector<std::filesystem::path> categories(const std::filesystem::path& root);
    std::string fingerprint(const std::filesystem::path& root);
} // namespace classifier
