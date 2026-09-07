import genesia.headless;
#if defined(GENESIA_HAS_EDITOR)
import genesia.editor;
#endif
import std;

int main(const int argc, char** argv) try {
    if (argc == 1 || std::string_view{argv[1]} == "--help") {
        std::println("Genesia {}\nUsage: genesia configuration.json [--gui]", GENESIA_VERSION);
        return 0;
    }
    const std::filesystem::path path{argv[1]};
    const auto configuration = genesia::read_configuration(path);
    if (argc > 2) {
        if (std::string_view{argv[2]} != "--gui") throw std::runtime_error{"Unknown command-line option"};
#if defined(GENESIA_HAS_EDITOR)
        genesia::editor::run(configuration, path);
        return 0;
#else
        throw std::runtime_error{"This build does not contain the Editor"};
#endif
    }
    genesia::headless::run(configuration);
    return 0;
} catch (const std::exception& error) {
    std::println(std::cerr, "Genesia: {}", error.what());
    return 1;
}
