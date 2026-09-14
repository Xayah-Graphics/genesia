import genesia.headless;
import genesia.io.files;
#if defined(GENESIA_HAS_EDITOR)
import genesia.editor;
#endif
import std;
int main(const int argc, char** argv) try {
    const genesia::files::Instance instance;
    std::vector<std::string_view> arguments{argv + 1, argv + argc};
    if (!arguments.empty() && arguments.front() == "--gui") {
#if defined(GENESIA_HAS_EDITOR)
        return genesia::editor::run(std::span{arguments}.subspan(1));
#else
        throw std::runtime_error{"This build does not contain the Editor"};
#endif
    }
    return genesia::headless::run(arguments);
} catch (const std::exception& error) {
    std::println(std::cerr, "Genesia: {}", error.what());
    return 1;
}
