module;
#include <nlohmann/json.hpp>
export module genesia.io.safetensors;
export import genesia.io.files;
import std;
export namespace genesia::files {
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
} // namespace genesia::files
