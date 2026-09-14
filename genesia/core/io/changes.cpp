module;
#if defined(_WIN32)
#include <Windows.h>
#else
#include <cerrno>
#include <poll.h>
#include <sys/inotify.h>
#include <unistd.h>
#endif
module genesia.io.changes;
import std;
namespace genesia::files {
    Changes::Changes(std::filesystem::path path) : directory{std::move(path)} {
#if defined(_WIN32)
        folder = CreateFileW(directory.c_str(), FILE_LIST_DIRECTORY, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, nullptr);
        if (folder == INVALID_HANDLE_VALUE) throw std::system_error{int(GetLastError()), std::system_category(), "Open dataset notifications"};
        event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!event) {
            CloseHandle(folder);
            throw std::system_error{int(GetLastError()), std::system_category(), "Create dataset notification"};
        }
        operation.hEvent = event;
        buffer.resize(64 * 1024);
        try {
            watch();
        } catch (...) {
            CloseHandle(event);
            CloseHandle(folder);
            throw;
        }
#else
        handle = inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
        if (handle == -1) throw std::system_error{errno, std::generic_category(), "Open dataset notifications"};
        std::vector<std::filesystem::path> folders{directory};
        for (const auto& entry : std::filesystem::recursive_directory_iterator{directory})
            if (entry.is_directory()) folders.push_back(entry.path());
        try {
            for (const auto& folder : folders) {
                const auto descriptor = inotify_add_watch(handle, folder.c_str(), IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_CLOSE_WRITE | IN_DELETE_SELF);
                if (descriptor == -1) throw std::system_error{errno, std::generic_category(), "Watch dataset directory"};
                watches[descriptor] = folder;
            }
        } catch (...) {
            close(handle);
            throw;
        }
#endif
    }
    Changes::~Changes() {
#if defined(_WIN32)
        if (reading) {
            CancelIoEx(folder, &operation);
            DWORD bytes;
            GetOverlappedResult(folder, &operation, &bytes, TRUE);
        }
        CloseHandle(event);
        CloseHandle(folder);
#else
        close(handle);
#endif
    }
    bool Changes::poll(const std::chrono::milliseconds timeout) {
        bool changed{};
#if defined(_WIN32)
        const auto wait = WaitForSingleObject(event, DWORD(timeout.count()));
        if (wait == WAIT_TIMEOUT) return false;
        if (wait == WAIT_FAILED) throw std::system_error{int(GetLastError()), std::system_category(), "Wait for datasets"};
        DWORD bytes;
        if (!GetOverlappedResult(folder, &operation, &bytes, FALSE)) throw std::system_error{int(GetLastError()), std::system_category(), "Read dataset changes"};
        reading = false;
        changed = !bytes;
        for (DWORD offset = 0; offset < bytes;) {
            const auto& entry = *reinterpret_cast<const FILE_NOTIFY_INFORMATION*>(buffer.data() + offset);
            const std::filesystem::path path{std::wstring_view{entry.FileName, entry.FileNameLength / sizeof(wchar_t)}};
            const auto name = path.filename();
            auto extension  = path.extension().wstring();
            std::ranges::transform(extension, extension.begin(), [](wchar_t c) { return std::towlower(c); });
            changed |= extension == L".png" || extension.empty() || name == L"concept.json" || name == L"state.json" || name == L"model.json";
            if (!entry.NextEntryOffset) break;
            offset += entry.NextEntryOffset;
        }
        watch();
#else
        pollfd descriptor{handle, POLLIN, 0};
        const auto wait = ::poll(&descriptor, 1, int(timeout.count()));
        if (wait == -1) {
            if (errno == EINTR) return false;
            throw std::system_error{errno, std::generic_category(), "Wait for datasets"};
        }
        if (!wait) return false;
        alignas(inotify_event) std::array<char, 64 * 1024> buffer;
        const auto bytes = read(handle, buffer.data(), buffer.size());
        if (bytes == -1) throw std::system_error{errno, std::generic_category(), "Read dataset changes"};
        for (std::size_t offset = 0; offset < std::size_t(bytes);) {
            const auto& entry = *reinterpret_cast<const inotify_event*>(buffer.data() + offset);
            offset += sizeof(entry) + entry.len;
            if (entry.mask & IN_Q_OVERFLOW) throw std::runtime_error{"Dataset notification queue overflow"};
            if (entry.mask & IN_IGNORED) {
                watches.erase(entry.wd);
                continue;
            }
            const auto path = watches.at(entry.wd) / (entry.len ? entry.name : "");
            if ((entry.mask & IN_ISDIR) && (entry.mask & (IN_CREATE | IN_MOVED_TO))) {
                std::vector<std::filesystem::path> folders{path};
                for (const auto& child : std::filesystem::recursive_directory_iterator{path})
                    if (child.is_directory()) folders.push_back(child.path());
                for (const auto& folder : folders) {
                    const auto next = inotify_add_watch(handle, folder.c_str(), IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_CLOSE_WRITE | IN_DELETE_SELF);
                    if (next == -1) throw std::system_error{errno, std::generic_category(), "Watch dataset directory"};
                    watches[next] = folder;
                }
            }
            auto extension = path.extension().string();
            std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            changed |= (entry.mask & IN_ISDIR) || extension == ".png" || path.filename() == "concept.json" || path.filename() == "state.json" || path.filename() == "model.json";
        }
#endif
        return changed;
    }
#if defined(_WIN32)
    void Changes::watch() {
        ResetEvent(event);
        if (!ReadDirectoryChangesW(folder, buffer.data(), DWORD(buffer.size()), TRUE, FILE_NOTIFY_CHANGE_DIR_NAME | FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_SIZE | FILE_NOTIFY_CHANGE_LAST_WRITE, nullptr, &operation, nullptr)) throw std::system_error{int(GetLastError()), std::system_category(), "Watch datasets"};
        reading = true;
    }
#endif

} // namespace genesia::files
