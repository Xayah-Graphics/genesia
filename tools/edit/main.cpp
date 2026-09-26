#if defined(EDIT_HAS_EDITOR)
#include <Windows.h>
#include <cstdlib>
#endif
import tools.application;
import edit.headless;
import tools.files;
#if defined(EDIT_HAS_EDITOR)
import edit.editor;
#endif
import std;
#if defined(EDIT_HAS_EDITOR)
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    const int argc        = __argc;
    char** argv           = __argv;
    constexpr bool editor = true;
#else
int main(const int argc, char** argv) {
    constexpr bool editor = false;
#endif
    return tools::application::run("Edit", argc, argv, editor, [](const bool editor, const std::span<const std::string_view> arguments) {
        const tools::files::Instance instance{"edit"};
#if defined(EDIT_HAS_EDITOR)
        if (editor) return edit::editor::run(arguments);
#endif
        return edit::headless::run(arguments);
    });
}
