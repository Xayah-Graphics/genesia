module;
#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <nlohmann/json.hpp>
module genesia.io.safetensors;
import std;
import genesia.io.files;
namespace genesia::files {
    Mapping::Mapping(const std::filesystem::path& path) {
        size = std::filesystem::file_size(path);
#if defined(_WIN32)
        HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw std::system_error(int(GetLastError()), std::system_category(), "Open model/cache");
        HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        data           = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(mapping);
        CloseHandle(file);
        if (!data) throw std::system_error(int(GetLastError()), std::system_category(), "Map model/cache");
#elif defined(__linux__)
        const int file = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (file < 0) throw std::system_error(errno, std::generic_category(), "Open model/cache");
        data              = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file, 0);
        const int failure = errno;
        ::close(file);
        if (data == MAP_FAILED) throw std::system_error(failure, std::generic_category(), "Map model/cache");
#endif
    }
    Mapping::~Mapping() {
#if defined(_WIN32)
        UnmapViewOfFile(data);
#elif defined(__linux__)
        ::munmap(data, size);
#endif
    }
    SafeFile::SafeFile(const std::filesystem::path& path) : mapping(path) {
        auto length = *static_cast<std::uint64_t*>(mapping.data);
        base        = static_cast<const std::byte*>(mapping.data) + 8 + length;
        header      = nlohmann::json::parse(static_cast<const char*>(mapping.data) + 8, static_cast<const char*>(mapping.data) + 8 + length);
    }
} // namespace genesia::files
