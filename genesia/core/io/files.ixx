module;
#include <nlohmann/json.hpp>
export module genesia.io.files;
export import genesia.project;
import std;

export namespace genesia::files {
    struct Instance final {
        Instance();
        ~Instance();
        Instance(const Instance&)            = delete;
        Instance& operator=(const Instance&) = delete;

    private:
        std::intptr_t handle{};
    };
    std::string utf8(const std::filesystem::path& path);
    std::filesystem::path path(std::string_view text);
    nlohmann::json read_json(const std::filesystem::path& path);
    void write_json(const std::filesystem::path& path, const nlohmann::json& value);
    std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path);
    void write_bytes(const std::filesystem::path& path, std::span<const std::uint8_t> bytes);
    std::string digest(const std::filesystem::path& path);
    void publish(const std::filesystem::path& source, const std::filesystem::path& destination);
    void move(const std::filesystem::path& source, const std::filesystem::path& destination);
} // namespace genesia::files
