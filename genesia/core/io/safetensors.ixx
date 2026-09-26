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
} // namespace genesia::files
