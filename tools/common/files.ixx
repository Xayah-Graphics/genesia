module;
#include <nlohmann/json.hpp>
export module tools.files;
import std;

export namespace tools::files {
    struct Instance final {
        explicit Instance(std::string_view application);
        ~Instance();
        Instance(const Instance&)            = delete;
        Instance& operator=(const Instance&) = delete;

    private:
        std::intptr_t handle{};
    };
    std::string utf8(const std::filesystem::path& path);
    std::filesystem::path path(std::string_view text);
    std::string read_text(const std::filesystem::path& path);
    nlohmann::json read_json(const std::filesystem::path& path);
    void write_json(const std::filesystem::path& path, const nlohmann::json& value);
    std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path);
    void write_bytes(const std::filesystem::path& path, std::span<const std::uint8_t> bytes);
    std::string digest(const std::filesystem::path& path);
    void publish(const std::filesystem::path& source, const std::filesystem::path& destination);
    void move(const std::filesystem::path& source, const std::filesystem::path& destination);
} // namespace tools::files
