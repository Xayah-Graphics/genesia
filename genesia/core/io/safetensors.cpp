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
    void save_tensors(const std::filesystem::path& path, const std::map<std::string, HostTensor>& tensors, const nlohmann::json& metadata) {
        std::filesystem::create_directories(path.parent_path());
        nlohmann::json header;
        header["__metadata__"] = metadata;
        std::size_t offset     = 0;
        for (const auto& [name, tensor] : tensors) {
            header[name] = {{"dtype", "F32"}, {"shape", tensor.shape}, {"data_offsets", {offset, offset + tensor.values.size() * 4}}};
            offset += tensor.values.size() * 4;
        }
        std::string text = header.dump();
        text.resize((text.size() + 7) / 8 * 8, ' ');
        std::uint64_t length = text.size();
        auto temporary       = path;
        temporary += ".tmp";
        std::ofstream file(temporary, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&length), 8);
        file.write(text.data(), text.size());
        for (const auto& [name, tensor] : tensors) file.write(reinterpret_cast<const char*>(tensor.values.data()), tensor.values.size() * 4);
        file.close();
        if (!file) throw std::runtime_error("Cannot write checkpoint");
        files::publish(temporary, path);
    }
} // namespace genesia::files
