#if defined(_WIN32)
#include <Windows.h>
#include <cstdio>
#include <io.h>
#endif
#if defined(GENESIA_HAS_EDITOR)
#include <cstdlib>
#endif
import genesia.headless;
import genesia.lan;
import genesia.runtime.diagnostics;
#if defined(GENESIA_HAS_EDITOR)
import genesia.editor;
#endif
import std;

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
            // Preserve inherited files and pipes; attach only unbound CRT streams.
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

#if defined(GENESIA_HAS_EDITOR)
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    const int argc = __argc;
    char** argv    = __argv;
    bool editor    = true;
#else
int main(const int argc, char** argv) {
    bool editor = false;
#endif
    bool logging{};
    try {
        const std::vector<std::string_view> values{argv + 1, argv + argc};
        std::span<const std::string_view> arguments{values};
        const bool lan = !arguments.empty() && arguments.front() == "--lan";
        if (lan || (!arguments.empty() && arguments.front() == "--headless")) {
            editor    = false;
            arguments = arguments.subspan(1);
        }
#if defined(_WIN32)
        if (!editor) attach_console();
#endif
        genesia::diagnostics::initialize(editor);
        logging = true;
        int result;
#if defined(GENESIA_HAS_EDITOR)
        if (editor) result = genesia::editor::run(arguments);
        else
#endif
            result = lan ? genesia::lan::run(arguments) : genesia::headless::run(arguments);
        genesia::diagnostics::record("process.exit", std::format("code={}", result));
        return result;
    } catch (const std::exception& error) {
        if (logging) genesia::diagnostics::record("main.exception", error.what());
#if defined(_WIN32)
        if (editor) MessageBoxA(nullptr, error.what(), "Genesia", MB_OK | MB_ICONERROR);
        else
#endif
            std::println(std::cerr, "Genesia: {}", error.what());
        return 1;
    }
}
