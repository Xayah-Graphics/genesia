module;
#if defined(_WIN32)
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <csignal>
module genesia.runtime.diagnostics;
import std;

namespace genesia::diagnostics {
    std::filesystem::path log_path;
    namespace {
        // Keep the log open until process exit, including static destruction and fatal handlers.
#if defined(_WIN32)
        HANDLE file{INVALID_HANDLE_VALUE};
#else
        int file{-1};
#endif

        bool append(std::string_view bytes) noexcept {
            while (!bytes.empty()) {
#if defined(_WIN32)
                DWORD written{};
                if (!WriteFile(file, bytes.data(), static_cast<DWORD>(std::min(bytes.size(), std::size_t{MAXDWORD})), &written, nullptr) || !written) return false;
#else
                const auto written = write(file, bytes.data(), bytes.size());
                if (written < 0 && errno == EINTR) continue;
                if (written <= 0) return false;
#endif
                bytes.remove_prefix(static_cast<std::size_t>(written));
            }
            return true;
        }

        void fatal(const char* event, const char* message) noexcept {
            // No application locks, heap formatting, or GPU calls on the crash path.
            char line[256];
            const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
#if defined(_WIN32)
            const auto thread = static_cast<unsigned long>(GetCurrentThreadId());
#else
            const auto thread = static_cast<unsigned long>(syscall(SYS_gettid));
#endif
            std::snprintf(line, sizeof(line), "\nFATAL epoch_ms=%lld thread=%lu ", static_cast<long long>(milliseconds), thread);
            append(line);
            append(event);
            append(": ");
            append(message);
            append("\n");
#if defined(_WIN32)
            std::snprintf(line, sizeof(line), "exe_base=%p\n", static_cast<void*>(GetModuleHandleW(nullptr)));
            append(line);
            void* frames[64];
            const auto count = CaptureStackBackTrace(0, 64, frames, nullptr);
            for (unsigned int i = 0; i < count; ++i) {
                std::snprintf(line, sizeof(line), "thread=%lu frame=%u address=%p\n", thread, i, frames[i]);
                append(line);
            }
            FlushFileBuffers(file);
#else
            fsync(file);
#endif
        }
    } // namespace

    void initialize(const bool editor) {
        const std::filesystem::path directory{std::filesystem::path{GENESIA_RUNTIME_DIRECTORY} / "logs"};
        std::filesystem::create_directories(directory);
#if defined(_WIN32)
        const auto process = GetCurrentProcessId();
#else
        const auto process = getpid();
#endif
        log_path = directory / std::format("genesia-{:%Y%m%d-%H%M%S}-{}.log", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()), process);
#if defined(_WIN32)
        file = CreateFileW(log_path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Create Genesia diagnostic log"};
#else
        file = open(log_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC, 0600);
        if (file == -1) throw std::system_error{errno, std::generic_category(), "Create Genesia diagnostic log"};
#endif
        std::set_terminate([] {
            const auto exception = std::current_exception();
            if (exception) {
                try {
                    std::rethrow_exception(exception);
                } catch (const std::exception& error) {
                    fatal("std::terminate", error.what());
                } catch (...) {
                    fatal("std::terminate", "non-standard C++ exception");
                }
            } else fatal("std::terminate", "no active C++ exception");
            std::abort();
        });
        std::signal(SIGABRT, [](int) { fatal("SIGABRT", "abort requested"); });
#if defined(_WIN32)
        SetUnhandledExceptionFilter([](EXCEPTION_POINTERS* exception) -> LONG {
            const auto& fault = *exception->ExceptionRecord;
            char detail[256];
            std::snprintf(detail, sizeof(detail), "code=0x%08lx address=%p parameters=%lu first=0x%llx second=0x%llx", static_cast<unsigned long>(fault.ExceptionCode), fault.ExceptionAddress, static_cast<unsigned long>(fault.NumberParameters), fault.NumberParameters > 0 ? static_cast<unsigned long long>(fault.ExceptionInformation[0]) : 0ull, fault.NumberParameters > 1 ? static_cast<unsigned long long>(fault.ExceptionInformation[1]) : 0ull);
            fatal("unhandled SEH", detail);
            return EXCEPTION_CONTINUE_SEARCH;
        });
#endif
        record("process.start", std::format("version={} build={} {} mode={} pid={}", GENESIA_VERSION, __DATE__, __TIME__, editor ? "editor" : "headless", process));
    }

    void record(const std::string_view event, const std::string_view detail, const std::source_location source) {
#if defined(_WIN32)
        const auto thread = GetCurrentThreadId();
#else
        const auto thread = syscall(SYS_gettid);
#endif
        const auto line = std::format("{:%FT%TZ} thread={} {} {} [{}:{}]\n", std::chrono::floor<std::chrono::milliseconds>(std::chrono::system_clock::now()), thread, event, detail, source.file_name(), source.line());
        if (!append(line)) throw std::runtime_error{"Write Genesia diagnostic log failed"};
    }
} // namespace genesia::diagnostics
