module;
#include <Windows.h>

#include <GLFW/glfw3.h>
module genesia.editor.gallery;
import std;

namespace genesia::editor {
    Gallery::Gallery(std::filesystem::path directory, std::shared_ptr<const prompt::Catalog> catalog) : directory{std::filesystem::absolute(directory)}, catalog{std::move(catalog)} {
        wake.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
        if (!wake) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Create gallery event"};
        worker = std::jthread{[this](const std::stop_token stop) { read(stop); }};
    }

    Gallery::~Gallery() {
        worker.request_stop();
        SetEvent(wake.get());
        worker.join();
    }

    std::vector<Gallery::Result> Gallery::receive() {
        const std::lock_guard lock{mutex};
        if (index) {
            files = std::move(*index);
            index.reset();
            positions.clear();
            for (std::size_t i = 0; i < files.size(); ++i) positions.emplace(files[i].id, i);
            ++revision;
            ready = true;
        }
        if (!failure.empty()) error = std::exchange(failure, {});
        pending       = false;
        auto received = std::exchange(results, {});
        SetEvent(wake.get());
        return received;
    }

    std::uint64_t Gallery::select(const File& file) {
        const std::lock_guard lock{mutex};
        requested_image = file;
        ++ticket;
        SetEvent(wake.get());
        return ticket;
    }

    void Gallery::cancel() {
        const std::lock_guard lock{mutex};
        requested_image.reset();
        ++ticket;
        SetEvent(wake.get());
    }

    void Gallery::thumbnails(std::vector<File> files) {
        const std::lock_guard lock{mutex};
        if (requested_thumbnails == files) return;
        requested_thumbnails = std::move(files);
        SetEvent(wake.get());
    }

    void Gallery::refresh() {
        const std::lock_guard lock{mutex};
        rescan = true;
        SetEvent(wake.get());
    }

    void Gallery::read(const std::stop_token stop) {
        try {
            const auto handle = CreateFileW(directory.c_str(), FILE_LIST_DIRECTORY, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, nullptr);
            if (handle == INVALID_HANDLE_VALUE) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Open gallery directory"};
            const std::unique_ptr<std::remove_pointer_t<HANDLE>, decltype(&CloseHandle)> folder{handle, CloseHandle};
            const std::unique_ptr<std::remove_pointer_t<HANDLE>, decltype(&CloseHandle)> changed{CreateEventW(nullptr, TRUE, FALSE, nullptr), CloseHandle};
            if (!changed) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Create directory notification"};
            alignas(8) std::array<std::byte, 64 * 1024> notifications{}, listing{};
            OVERLAPPED operation{};
            operation.hEvent = changed.get();
            struct PendingRead final {
                HANDLE folder;
                OVERLAPPED& operation;
                bool active{};
                ~PendingRead() {
                    if (!active) return;
                    CancelIoEx(folder, &operation);
                    DWORD bytes;
                    GetOverlappedResult(folder, &operation, &bytes, TRUE);
                }
            } notification{handle, operation};
            struct Cached final {
                File file;
                std::shared_ptr<const Image> image;
                std::uint64_t touched{};
                std::string error;
            };
            std::map<std::uint64_t, Cached> cache;
            std::size_t cache_bytes{};
            std::uint64_t clock{}, completed_ticket{};
            std::optional<File> decoded_file;
            std::shared_ptr<const Image> decoded;
            std::vector<File> delivered;
            auto scan_at = std::chrono::steady_clock::time_point::max();
            for (;;) {
                if (stop.stop_requested()) break;
                if (!notification.active) {
                    ResetEvent(changed.get());
                    if (!ReadDirectoryChangesW(handle, notifications.data(), static_cast<DWORD>(notifications.size()), FALSE, FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_SIZE | FILE_NOTIFY_CHANGE_LAST_WRITE, nullptr, &operation, nullptr)) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Watch gallery directory"};
                    notification.active = true;
                }
                bool scan;
                {
                    const std::lock_guard lock{mutex};
                    scan = std::exchange(rescan, false);
                }
                if (scan || std::chrono::steady_clock::now() >= scan_at) {
                    std::vector<File> found;
                    bool first = true;
                    for (;;) {
                        if (!GetFileInformationByHandleEx(handle, first ? FileIdBothDirectoryRestartInfo : FileIdBothDirectoryInfo, listing.data(), static_cast<DWORD>(listing.size()))) {
                            if (GetLastError() == ERROR_NO_MORE_FILES) break;
                            throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Read gallery directory"};
                        }
                        first = false;
                        for (std::size_t offset = 0;;) {
                            const auto& entry = *reinterpret_cast<const FILE_ID_BOTH_DIR_INFO*>(listing.data() + offset);
                            const std::filesystem::path name{std::wstring_view{entry.FileName, entry.FileNameLength / sizeof(wchar_t)}};
                            const auto extension = name.extension().wstring();
                            if (!(entry.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) && _wcsicmp(extension.c_str(), L".png") == 0) found.push_back({static_cast<std::uint64_t>(entry.FileId.QuadPart), static_cast<std::uint64_t>(entry.LastWriteTime.QuadPart), static_cast<std::uint64_t>(entry.EndOfFile.QuadPart), directory / name});
                            if (!entry.NextEntryOffset) break;
                            offset += entry.NextEntryOffset;
                        }
                    }
                    std::ranges::sort(found, [](const File& a, const File& b) { return std::tie(a.modified, a.path) < std::tie(b.modified, b.path); });
                    {
                        const std::lock_guard lock{mutex};
                        index   = std::move(found);
                        pending = true;
                    }
                    glfwPostEmptyEvent();
                    scan_at = std::chrono::steady_clock::time_point::max();
                }
                std::optional<File> selected;
                std::uint64_t selected_ticket;
                std::vector<File> wanted;
                bool room;
                {
                    const std::lock_guard lock{mutex};
                    selected_ticket = ticket;
                    if (ticket != completed_ticket) selected = requested_image;
                    wanted = requested_thumbnails;
                    room   = results.size() < 4;
                }
                std::erase_if(delivered, [&](const File& file) { return !std::ranges::contains(wanted, file); });
                std::optional<File> task = selected;
                if (!task && room) {
                    const auto missing = std::ranges::find_if(wanted, [&](const File& file) { return !std::ranges::contains(delivered, file); });
                    if (missing != wanted.end()) task = *missing;
                }
                if (task) {
                    Result result{*task};
                    result.ticket = selected ? selected_ticket : 0;
                    try {
                        auto cached = cache.find(task->id);
                        if (selected || cached == cache.end() || cached->second.file != *task) {
                            if (!decoded_file || *decoded_file != *task) {
                                decoded      = std::make_shared<Image>(read_image(task->path));
                                decoded_file = *task;
                            }
                            if (selected) {
                                result.image  = decoded;
                                result.record = read_record(task->path, catalog);
                            } else result.image = std::make_shared<Image>(thumbnail(decoded->pixels, decoded->width, decoded->height));
                        } else {
                            result.image = cached->second.image;
                            result.error = cached->second.error;
                        }
                    } catch (const std::exception& error) {
                        result.error = error.what();
                    }
                    if (!selected) {
                        auto& cached = cache[task->id];
                        if (cached.image) cache_bytes -= cached.image->pixels.size();
                        cached = {*task, result.image, ++clock, result.error};
                        if (cached.image) cache_bytes += cached.image->pixels.size();
                        while (cache_bytes > 16 * 1024 * 1024 || cache.size() > 256) {
                            const auto oldest = std::ranges::min_element(cache, {}, [](const auto& pair) { return pair.second.touched; });
                            if (oldest->second.image) cache_bytes -= oldest->second.image->pixels.size();
                            cache.erase(oldest);
                        }
                        delivered.push_back(*task);
                    } else completed_ticket = selected_ticket;
                    {
                        const std::lock_guard lock{mutex};
                        if ((selected && selected_ticket == ticket) || (!selected && std::ranges::contains(requested_thumbnails, *task))) {
                            results.push_back(std::move(result));
                            pending = true;
                        }
                    }
                    glfwPostEmptyEvent();
                }
                const auto now      = std::chrono::steady_clock::now();
                const DWORD timeout = task ? 0 : scan_at == std::chrono::steady_clock::time_point::max() ? INFINITE : static_cast<DWORD>(std::max<std::int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(scan_at - now).count()));
                const std::array handles{wake.get(), changed.get()};
                const DWORD signaled = WaitForMultipleObjects(static_cast<DWORD>(handles.size()), handles.data(), FALSE, timeout);
                if (signaled == WAIT_FAILED) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Wait for gallery work"};
                if (signaled == WAIT_OBJECT_0 + 1) {
                    DWORD bytes;
                    const BOOL success  = GetOverlappedResult(handle, &operation, &bytes, FALSE);
                    notification.active = false;
                    if (!success && GetLastError() != ERROR_NOTIFY_ENUM_DIR) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Read directory notification"};
                    scan_at = std::chrono::steady_clock::now() + std::chrono::milliseconds{150};
                }
            }
        } catch (const std::exception& error) {
            const std::lock_guard lock{mutex};
            failure = error.what();
            pending = true;
            glfwPostEmptyEvent();
        }
    }
} // namespace genesia::editor
