module;
#include <nlohmann/json.hpp>
export module classifier.storage;
import std;
export namespace classifier {
    std::filesystem::path default_cache_directory();
    std::string path_utf8(const std::filesystem::path& path);
    struct Mapping {
        void* data       = nullptr;
        std::size_t size = 0;
        explicit Mapping(const std::filesystem::path& path);
        ~Mapping();
        Mapping(const Mapping&) = delete;
    };
    struct SafeFile {
        Mapping mapping;
        nlohmann::json header;
        const std::byte* base;
        explicit SafeFile(const std::filesystem::path& path);
    };
    struct HostTensor {
        std::vector<std::int64_t> shape;
        std::vector<float> values;
    };
    void save_tensors(const std::filesystem::path& path, const std::map<std::string, HostTensor>& tensors, const nlohmann::json& metadata);
    void write_json(const std::filesystem::path& path, const nlohmann::json& value);
} // namespace classifier
