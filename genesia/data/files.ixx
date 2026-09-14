module;
#include <nlohmann/json.hpp>
export module genesia.files;
import std;

export namespace genesia::files {
    std::string utf8(const std::filesystem::path& path);
    std::filesystem::path path(std::string_view text);
    nlohmann::json read_json(const std::filesystem::path& path);
    void write_json(const std::filesystem::path& path, const nlohmann::json& value);
    std::string digest(const std::filesystem::path& path);
    void publish(const std::filesystem::path& source, const std::filesystem::path& destination);
    void move(const std::filesystem::path& source, const std::filesystem::path& destination);
} // namespace genesia::files
