module;
#include <nlohmann/json.hpp>
export module genesia.io.files;
export import genesia.project;
import std;

export namespace genesia::files {
    std::string utf8(const std::filesystem::path& path);
    std::filesystem::path path(std::string_view text);
    std::string random_id();
    nlohmann::json read_json(const std::filesystem::path& path);
    void write_json(const std::filesystem::path& path, const nlohmann::json& value, int indent = -1, bool replace = true);
    std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path);
    void write_bytes(const std::filesystem::path& path, std::span<const std::uint8_t> bytes, bool replace = true);
    std::string digest(const std::filesystem::path& path);
} // namespace genesia::files
