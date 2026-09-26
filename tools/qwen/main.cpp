#if defined(QWEN_HAS_EDITOR)
#include <Windows.h>
#include <cstdlib>
#endif
import tools.application;
import qwen.headless;
import tools.files;
#if defined(QWEN_HAS_EDITOR)
import qwen.editor;
#endif
import std;

#if defined(QWEN_HAS_EDITOR)
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    const int argc        = __argc;
    char** argv           = __argv;
    constexpr bool editor = true;
#else
int main(const int argc, char** argv) {
    constexpr bool editor = false;
#endif
    return tools::application::run("Qwen", argc, argv, editor, [](const bool editor, const std::span<const std::string_view> arguments) {
        const tools::files::Instance instance{"qwen"};
#if defined(QWEN_HAS_EDITOR)
        if (editor) return qwen::editor::run(arguments);
#endif
        return qwen::headless::run(arguments);
    });
}
