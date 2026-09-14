module;
#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <nlohmann/json.hpp>
module genesia.files;
import genesia.hash;
import std;

namespace genesia::files {
    std::string utf8(const std::filesystem::path& path) {
        const auto text = path.generic_u8string();
        return {text.begin(), text.end()};
    }
    std::filesystem::path path(const std::string_view text) {
        return std::filesystem::path{std::u8string{text.begin(), text.end()}};
    }
    nlohmann::json read_json(const std::filesystem::path& path) {
        std::ifstream file{path};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        return nlohmann::json::parse(file);
    }
    void write_json(const std::filesystem::path& path, const nlohmann::json& value) {
        std::filesystem::create_directories(path.parent_path());
        auto temporary = path;
        temporary += ".part";
        std::ofstream file{temporary, std::ios::binary | std::ios::trunc};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << value.dump();
        file.close();
        publish(temporary, path);
    }
    std::string digest(const std::filesystem::path& path) {
        std::ifstream file{path, std::ios::binary};
        file.exceptions(std::ios::badbit);
        if (!file) throw std::runtime_error{"Cannot read " + utf8(path)};
        Sha256 hash;
        std::vector<unsigned char> buffer(1024 * 1024);
        while (file.read(reinterpret_cast<char*>(buffer.data()), buffer.size()) || file.gcount()) hash.update({buffer.data(), static_cast<std::size_t>(file.gcount())});
        if (!file.eof()) throw std::runtime_error{"Cannot read " + utf8(path)};
        std::string result;
        for (const auto byte : hash.finish()) result += std::format("{:02x}", byte);
        return result;
    }
    void publish(const std::filesystem::path& source, const std::filesystem::path& destination) {
#if defined(_WIN32)
        if (!MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Publish " + utf8(destination)};
#else
        std::filesystem::rename(source, destination);
#endif
    }
    void move(const std::filesystem::path& source, const std::filesystem::path& destination) {
        std::filesystem::create_directories(destination.parent_path());
#if defined(_WIN32)
        if (!MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH)) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Move " + utf8(source) + " -> " + utf8(destination)};
#else
        if (syscall(SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD, destination.c_str(), RENAME_NOREPLACE)) throw std::system_error{errno, std::system_category(), "Move " + utf8(source) + " -> " + utf8(destination)};
#endif
    }
} // namespace genesia::files
