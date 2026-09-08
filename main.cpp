import genesia.headless;
#if defined(GENESIA_HAS_EDITOR)
import genesia.editor;
#endif
import std;

int main(const int argc, char** argv) try {
    bool gui{}, named_preset{}, headless_options{};
    int count{1};
    std::optional<std::uint64_t> first_seed;
    std::filesystem::path prompt_file;
    std::string_view name = genesia::defaults::preset;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option{argv[i]};
        if (option == "--help") {
            std::println("Genesia {}\nHeadless: genesia [--preset NAME | --prompt-file FILE] [--count N] [--seed SEED]\nEditor:   genesia --gui [--preset NAME]\nDefaults: preset={}, count=1, random seed per image.\n--seed uses SEED, SEED+1, ... (uint64).\n--prompt-file uses the existing positive/negative fixed/groups preset format.\nSaved images are reported as JSON Lines on stdout; diagnostics go to stderr.", GENESIA_VERSION, genesia::defaults::preset);
            return 0;
        }
        if (option == "--gui") gui = true;
        else if (option == "--preset" && i + 1 < argc) {
            name = argv[++i];
            named_preset = true;
        } else if (option == "--prompt-file" && i + 1 < argc) {
            prompt_file = argv[++i];
            headless_options = true;
        } else if (option == "--count" && i + 1 < argc) {
            const std::string_view value{argv[++i]};
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || count <= 0) throw std::invalid_argument{"--count requires a positive integer"};
            headless_options = true;
        } else if (option == "--seed" && i + 1 < argc) {
            const std::string_view value{argv[++i]};
            std::uint64_t seed{};
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), seed);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) throw std::invalid_argument{"--seed requires a uint64 integer"};
            first_seed = seed;
            headless_options = true;
        } else throw std::runtime_error{std::format("Unknown or incomplete command-line option: {}", option)};
    }
    if (named_preset && !prompt_file.empty()) throw std::invalid_argument{"Choose either --preset or --prompt-file"};
    if (gui && headless_options) throw std::invalid_argument{"--prompt-file, --count and --seed are headless options"};
#if !defined(GENESIA_HAS_EDITOR)
    if (gui) throw std::runtime_error{"This build does not contain the Editor"};
#endif
    auto catalog = std::make_shared<const genesia::prompt::Catalog>();
    auto preset = prompt_file.empty() ? genesia::prompt::read_preset(name, *catalog) : genesia::prompt::Preset{prompt_file.stem().string(), genesia::prompt::read_prompt(prompt_file, *catalog)};
#if defined(GENESIA_HAS_EDITOR)
    if (gui) {
        genesia::editor::run(std::move(preset), std::move(catalog));
        return 0;
    }
#endif
    genesia::headless::run(preset, catalog, count, first_seed);
    return 0;
} catch (const std::exception& error) {
    std::println(std::cerr, "Genesia: {}", error.what());
    return 1;
}
