module;
#include <Windows.h>
#include <GLFW/glfw3.h>
module genesia.editor.library;
import std;

namespace genesia::editor {
    Library::Library(std::shared_ptr<const prompt::Catalog> catalog, Renderer& renderer) : catalog{std::move(catalog)}, renderer{renderer} {
        std::filesystem::create_directories(dataset::raw);
        wake.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
        if (!wake) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Create library event"};
        worker = std::jthread{[this](const std::stop_token stop) { read(stop); }};
    }

    Library::~Library() {
        worker.request_stop();
        SetEvent(wake.get());
        worker.join();
        for (const auto& [sha, texture] : textures)
            if (texture.texture) renderer.retire(texture.texture);
    }

    void Library::receive() {
        std::vector<Decoded> decoded;
        {
            const std::lock_guard lock{mutex};
            if (index) {
                roots          = std::move(*index);
                concepts       = std::move(concept_updates);
                classifiers    = std::move(classifier_updates);
                concept_errors = std::move(concept_error_updates);
                index.reset();
                ++revision;
                ready = true;
            }
            if (!failure.empty()) {
                error = std::exchange(failure, {});
                ready = false;
                for (auto& root : roots) root.ready = false;
            }
            pending = false;
            decoded = std::exchange(results, {});
        }
        for (auto& result : decoded) {
            auto& cached = textures[result.file.sha];
            if (cached.texture) renderer.retire(cached.texture);
            texture_bytes -= cached.bytes;
            cached = {std::move(result.file), std::move(result.record), 0, ++clock, 0, std::move(result.error)};
            if (cached.error.empty()) {
                cached.texture = renderer.upload(result.image);
                cached.bytes   = static_cast<std::size_t>(result.image.width) * result.image.height * 4;
                texture_bytes += cached.bytes;
            }
        }
        SetEvent(wake.get());
    }

    void Library::request(std::vector<dataset::File> files) {
        std::set<std::string> pinned;
        std::vector<dataset::File> missing;
        for (const auto& file : files) {
            if (!pinned.insert(file.sha).second) continue;
            const auto cached = textures.find(file.sha);
            if (cached == textures.end()) missing.push_back(file);
            else cached->second.touched = ++clock;
        }
        while (texture_bytes > 512ULL * 1024 * 1024) {
            auto oldest = textures.end();
            for (auto entry = textures.begin(); entry != textures.end(); ++entry)
                if (!pinned.contains(entry->first) && (oldest == textures.end() || entry->second.touched < oldest->second.touched)) oldest = entry;
            if (oldest == textures.end()) break;
            if (oldest->second.texture) renderer.retire(oldest->second.texture);
            texture_bytes -= oldest->second.bytes;
            textures.erase(oldest);
        }
        const std::lock_guard lock{mutex};
        if (requested == missing) return;
        requested = std::move(missing);
        SetEvent(wake.get());
    }

    void Library::refresh() {
        const std::lock_guard lock{mutex};
        rescan = true;
        SetEvent(wake.get());
    }

    void Library::read(const std::stop_token stop) {
        try {
            const auto handle = CreateFileW(dataset::directory.c_str(), FILE_LIST_DIRECTORY, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, nullptr);
            if (handle == INVALID_HANDLE_VALUE) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Open datasets directory"};
            const std::unique_ptr<std::remove_pointer_t<HANDLE>, decltype(&CloseHandle)> folder{handle, CloseHandle};
            const std::unique_ptr<std::remove_pointer_t<HANDLE>, decltype(&CloseHandle)> changed{CreateEventW(nullptr, TRUE, FALSE, nullptr), CloseHandle};
            if (!changed) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Create dataset notification"};
            std::vector<std::byte> notifications(64 * 1024);
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
            dataset::Index scanner;
            std::set<std::string> delivered;
            auto scan_at = std::chrono::steady_clock::time_point::max();
            for (;;) {
                if (stop.stop_requested()) break;
                if (!notification.active) {
                    ResetEvent(changed.get());
                    if (!ReadDirectoryChangesW(handle, notifications.data(), static_cast<DWORD>(notifications.size()), TRUE, FILE_NOTIFY_CHANGE_DIR_NAME | FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_SIZE | FILE_NOTIFY_CHANGE_LAST_WRITE, nullptr, &operation, nullptr)) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Watch datasets directory"};
                    notification.active = true;
                }
                bool scan;
                {
                    const std::lock_guard lock{mutex};
                    scan = std::exchange(rescan, false);
                }
                if (scan || std::chrono::steady_clock::now() >= scan_at) {
                    const dataset::Lock files_lock{"image-moves"};
                    scanner.scan();
                    std::map<std::string, dataset::Concept> next;
                    std::map<std::string, classifier::TrainingData> next_classifiers;
                    std::map<std::string, std::string> errors;
                    for (const auto& root : scanner.roots)
                        for (const auto& collection : root.concepts) {
                            try {
                                auto assigned        = dataset::read_concept(collection.key);
                                next[collection.key] = assigned;
                                if (assigned.type == dataset::ConceptType::classifier) next_classifiers[collection.key] = classifier::inspect(assigned, root);
                            } catch (const std::exception& error) {
                                errors[collection.key] = error.what();
                            }
                        }
                    {
                        const std::lock_guard lock{mutex};
                        index                 = std::move(scanner.roots);
                        concept_updates       = std::move(next);
                        classifier_updates    = std::move(next_classifiers);
                        concept_error_updates = std::move(errors);
                        pending               = true;
                    }
                    glfwPostEmptyEvent();
                    scan_at = std::chrono::steady_clock::time_point::max();
                }
                std::optional<dataset::File> task;
                {
                    const std::lock_guard lock{mutex};
                    std::erase_if(delivered, [&](const auto& sha) { return std::ranges::find(requested, sha, &dataset::File::sha) == requested.end(); });
                    if (results.size() < 2) {
                        const auto next = std::ranges::find_if(requested, [&](const auto& file) { return !delivered.contains(file.sha); });
                        if (next != requested.end()) task = *next;
                    }
                }
                if (task) {
                    Decoded result{*task};
                    try {
                        result.record = read_record(task->path, catalog);
                        result.image  = read_image(task->path);
                    } catch (const std::exception& error) {
                        result.error = std::format("{}: {}", task->path.string(), error.what());
                    }
                    delivered.insert(task->sha);
                    {
                        const std::lock_guard lock{mutex};
                        results.push_back(std::move(result));
                        pending = true;
                    }
                    glfwPostEmptyEvent();
                }
                const std::array handles{wake.get(), changed.get()};
                const auto remaining = scan_at == std::chrono::steady_clock::time_point::max() ? INFINITE : static_cast<DWORD>(std::max<std::int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(scan_at - std::chrono::steady_clock::now()).count()));
                const auto event     = WaitForMultipleObjects(static_cast<DWORD>(handles.size()), handles.data(), FALSE, task ? 0 : remaining);
                if (event == WAIT_FAILED) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Wait for dataset changes"};
                if (event == WAIT_OBJECT_0 + 1) {
                    DWORD bytes;
                    if (!GetOverlappedResult(handle, &operation, &bytes, FALSE)) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Read dataset changes"};
                    notification.active = false;
                    scan_at             = std::chrono::steady_clock::now() + std::chrono::milliseconds{120};
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
