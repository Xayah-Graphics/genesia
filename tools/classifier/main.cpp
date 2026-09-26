#if defined(CLASSIFIER_HAS_EDITOR)
#include <Windows.h>
#include <cstdlib>
#endif
import tools.application;
import classifier.headless;
import tools.files;
#if defined(CLASSIFIER_HAS_EDITOR)
import classifier.editor;
#endif
import std;
#if defined(CLASSIFIER_HAS_EDITOR)
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    const int argc        = __argc;
    char** argv           = __argv;
    constexpr bool editor = true;
#else
int main(const int argc, char** argv) {
    constexpr bool editor = false;
#endif
    return tools::application::run("Classifier", argc, argv, editor, [](const bool editor, const std::span<const std::string_view> arguments) {
        const tools::files::Instance instance{"classifier"};
#if defined(CLASSIFIER_HAS_EDITOR)
        if (editor) return classifier::editor::run(arguments);
#endif
        return classifier::headless::run(arguments);
    });
}
