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
import std;

namespace genesia::files {
    struct Sha256 final {
        std::array<std::uint32_t, 8> state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::array<unsigned char, 64> buffer{};
        std::uint64_t bytes{};
        void update(std::span<const unsigned char> input);
        std::array<unsigned char, 32> finish();

    private:
        void compress(const unsigned char* input);
    };

    void Sha256::update(std::span<const unsigned char> input) {
        const std::size_t used = bytes % 64;
        bytes += input.size();
        if (used && !input.empty()) {
            const auto count = std::min(input.size(), 64 - used);
            std::memcpy(buffer.data() + used, input.data(), count);
            input = input.subspan(count);
            if (used + count < 64) return;
            compress(buffer.data());
        }
        while (input.size() >= 64) {
            compress(input.data());
            input = input.subspan(64);
        }
        std::ranges::copy(input, buffer.begin());
    }
    std::array<unsigned char, 32> Sha256::finish() {
        std::size_t used = bytes % 64;
        buffer[used++]   = 0x80;
        if (used > 56) {
            std::fill(buffer.begin() + used, buffer.end(), 0);
            compress(buffer.data());
            used = 0;
        }
        std::fill(buffer.begin() + used, buffer.begin() + 56, 0);
        const std::uint64_t bits = bytes * 8;
        for (int i = 0; i < 8; ++i) buffer[63 - i] = static_cast<unsigned char>(bits >> (8 * i));
        compress(buffer.data());
        std::array<unsigned char, 32> digest{};
        for (int i = 0; i < 32; ++i) digest[i] = static_cast<unsigned char>(state[i / 4] >> (24 - 8 * (i % 4)));
        return digest;
    }
    void Sha256::compress(const unsigned char* input) {
        static constexpr std::array<std::uint32_t, 64> constants{0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::array<std::uint32_t, 64> words{};
        for (int i = 0; i < 16; ++i) words[i] = (std::uint32_t(input[4 * i]) << 24) | (std::uint32_t(input[4 * i + 1]) << 16) | (std::uint32_t(input[4 * i + 2]) << 8) | input[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const auto a = words[i - 15], b = words[i - 2];
            words[i] = words[i - 16] + (std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3)) + words[i - 7] + (std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10));
        }
        auto [a, b, c, d, e, f, g, h] = state;
        for (int i = 0; i < 64; ++i) {
            const auto first  = h + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) + ((e & f) ^ (~e & g)) + constants[i] + words[i];
            const auto second = (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h                 = g;
            g                 = f;
            f                 = e;
            e                 = d + first;
            d                 = c;
            c                 = b;
            b                 = a;
            a                 = first + second;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    Instance::Instance() {
        const auto path = std::filesystem::temp_directory_path() / "genesia-instance.lock";
#if defined(_WIN32)
        const auto file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Open Genesia instance lock"};
        OVERLAPPED operation{};
        if (!LockFileEx(file, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &operation)) {
            const auto error = GetLastError();
            CloseHandle(file);
            if (error == ERROR_LOCK_VIOLATION) throw std::runtime_error{"Genesia is already running"};
            throw std::system_error{static_cast<int>(error), std::system_category(), "Lock Genesia instance"};
        }
        handle = reinterpret_cast<std::intptr_t>(file);
#else
        handle = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
        if (handle == -1) throw std::system_error{errno, std::generic_category(), "Open Genesia instance lock"};
        if (flock(static_cast<int>(handle), LOCK_EX | LOCK_NB) == -1) {
            const auto error = errno;
            close(static_cast<int>(handle));
            if (error == EWOULDBLOCK) throw std::runtime_error{"Genesia is already running"};
            throw std::system_error{error, std::generic_category(), "Lock Genesia instance"};
        }
#endif
    }
    Instance::~Instance() {
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
