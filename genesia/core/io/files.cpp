module;
#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <nlohmann/json.hpp>
module genesia.io.files;
import genesia.io.hash;
import std;

namespace genesia::files {
    Lock::Lock(const std::string_view name, const bool wait, const std::filesystem::path& directory, const bool shared) {
        std::filesystem::create_directories(directory);
        const auto path = directory / std::format("{}.lock", name);
#if defined(_WIN32)
        const auto file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Open resource lock"};
        OVERLAPPED operation{};
        if (!LockFileEx(file, (shared ? 0 : LOCKFILE_EXCLUSIVE_LOCK) | (wait ? 0 : LOCKFILE_FAIL_IMMEDIATELY), 0, 1, 0, &operation)) {
            const auto error = GetLastError();
            CloseHandle(file);
            if (!wait && error == ERROR_LOCK_VIOLATION) return;
            throw std::system_error{static_cast<int>(error), std::system_category(), "Lock resource"};
        }
        handle = reinterpret_cast<std::intptr_t>(file);
#else
        handle = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
        if (handle == -1) throw std::system_error{errno, std::generic_category(), "Open resource lock"};
        if (flock(static_cast<int>(handle), (shared ? LOCK_SH : LOCK_EX) | (wait ? 0 : LOCK_NB)) == -1) {
            const auto error = errno;
            close(static_cast<int>(handle));
            if (!wait && error == EWOULDBLOCK) return;
            throw std::system_error{error, std::generic_category(), "Lock resource"};
        }
#endif
        acquired = true;
    }

    Lock::~Lock() {
        if (!acquired) return;
#if defined(_WIN32)
        CloseHandle(reinterpret_cast<HANDLE>(handle));
#else
        close(static_cast<int>(handle));
#endif
    }


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
    std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
        std::ifstream file{path, std::ios::binary | std::ios::ate};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        return bytes;
    }
    void write_bytes(const std::filesystem::path& path, const std::span<const std::uint8_t> bytes) {
        std::filesystem::create_directories(path.parent_path());
        auto temporary = path;
        temporary += ".part";
        std::ofstream file{temporary, std::ios::binary | std::ios::trunc};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
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
