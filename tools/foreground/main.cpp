#if defined(FOREGROUND_HAS_EDITOR)
#include <Windows.h>
#include <cstdlib>
#endif
import tools.application;
import foreground.headless;
import tools.files;
#if defined(FOREGROUND_HAS_EDITOR)
import foreground.editor;
#endif
import std;
#if defined(FOREGROUND_HAS_EDITOR)
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    const int argc        = __argc;
    char** argv           = __argv;
    constexpr bool editor = true;
#else
int main(const int argc, char** argv) {
    constexpr bool editor = false;
#endif
    return tools::application::run("Foreground", argc, argv, editor, [](const bool editor, const std::span<const std::string_view> arguments) {
        const tools::files::Instance instance{"foreground"};
#if defined(FOREGROUND_HAS_EDITOR)
        if (editor) return foreground::editor::run(arguments);
#endif
        return foreground::headless::run(arguments);
    });
}
