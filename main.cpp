import genesia.headless;
#if defined(GENESIA_HAS_EDITOR)
import genesia.editor;
#endif
import std;

int main(const int argc, char** argv) try {
    bool gui{};
    std::string_view name = genesia::defaults::preset;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option{argv[i]};
        if (option == "--help") {
            std::println("Genesia {}\nUsage: genesia [--gui] [--preset name]\nDefault prompt: {}", GENESIA_VERSION, genesia::defaults::preset);
            return 0;
        }
        if (option == "--gui") gui = true;
        else if (option == "--preset" && i + 1 < argc) name = argv[++i];
        else throw std::runtime_error{std::format("Unknown or incomplete command-line option: {}", option)};
    }
#if !defined(GENESIA_HAS_EDITOR)
    if (gui) throw std::runtime_error{"This build does not contain the Editor"};
#endif
    auto catalog = std::make_shared<const genesia::prompt::Catalog>();
    auto preset  = genesia::prompt::read_preset(name, *catalog);
#if defined(GENESIA_HAS_EDITOR)
    if (gui) {
        genesia::editor::run(std::move(preset), std::move(catalog));
        return 0;
    }
#endif
    genesia::headless::run(preset, catalog);
    return 0;
} catch (const std::exception& error) {
    std::println(std::cerr, "Genesia: {}", error.what());
    return 1;
}
