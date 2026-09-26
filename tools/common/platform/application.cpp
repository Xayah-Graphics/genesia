module;
#if defined(_WIN32)
#include <Windows.h>
#include <cstdio>
#include <io.h>
#endif
module tools.application;
import std;

namespace tools::application {
#if defined(_WIN32)
    namespace {
        void attach_console() {
            if (!AttachConsole(ATTACH_PARENT_PROCESS)) {
                const auto error = GetLastError();
                // Console builds are already attached; pipe-only launches have no parent console.
                if (error == ERROR_ACCESS_DENIED || error == ERROR_INVALID_HANDLE) return;
                throw std::system_error{static_cast<int>(error), std::system_category(), "Attach parent console"};
            }

            const std::array streams{stdin, stdout, stderr};
            constexpr std::array handles{STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE};
            for (std::size_t i = 0; i < streams.size(); ++i) {
                // Keep inherited files and pipes. Only unbound CRT streams need the console.
                if (_fileno(streams[i]) < 0) {
                    FILE* stream{};
                    const auto error = freopen_s(&stream, i == 0 ? "CONIN$" : "CONOUT$", i == 0 ? "r" : "w", streams[i]);
                    if (error) throw std::system_error{error, std::generic_category(), "Connect standard stream to console"};
                }
                if (!SetStdHandle(handles[i], reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(streams[i]))))) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Connect standard handle"};
            }
        }
    } // namespace
#endif

    int run(const char* name, const int argc, char** argv, bool editor, int (*execute)(bool, std::span<const std::string_view>)) {
        try {
            const std::vector<std::string_view> values{argv + 1, argv + argc};
            std::span<const std::string_view> arguments{values};
            if (!arguments.empty() && arguments.front() == "--headless") {
                editor    = false;
                arguments = arguments.subspan(1);
            }
#if defined(_WIN32)
            if (!editor) attach_console();
#endif
            return execute(editor, arguments);
        } catch (const std::exception& error) {
#if defined(_WIN32)
            // Each executable's manifest sets the active Windows code page to UTF-8.
            if (editor) MessageBoxA(nullptr, error.what(), name, MB_OK | MB_ICONERROR);
            else
#endif
                std::println(std::cerr, "{}: {}", name, error.what());
            return 1;
        }
    }
} // namespace tools::application
