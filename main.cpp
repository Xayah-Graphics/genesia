import genesia.headless;
#if defined(GENESIA_HAS_EDITOR)
import genesia.editor;
#endif
import std;

int main(const int argc, char** argv) try {
    bool gui{}, named_preset{}, headless_options{};
    genesia::headless::Options options;
    std::filesystem::path prompt_file;
    std::string_view name = genesia::defaults::preset;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option{argv[i]};
        const auto argument = [&]() -> std::string_view {
            if (i + 1 == argc) throw std::invalid_argument{std::format("Missing argument for {}", option)};
            return argv[++i];
        };
        const auto number = [&](auto& destination) {
            const auto text = argument();
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), destination);
            if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) throw std::invalid_argument{std::format("Invalid number for {}", option)};
        };
        if (option == "--help") {
            std::println(R"(Genesia {}
Headless: genesia [--preset NAME | --prompt-file FILE] [--count N] [--seed SEED]
Repaint:  genesia --source PNG --denoise VALUE [--count N] [--seed SEED]
          [--preset NAME | --prompt-file FILE]
Editor:   genesia --gui [--preset NAME]
Defaults: preset={}, count=1, random seed per image.
--seed uses SEED, SEED+1, ... (uint64).
Repaint inherits the source PNG prompt, steps and CFG; a preset overrides its prompt.
--denoise is required for whole-image Repaint, in [0,1]; 0 preserves the source pixels.
Repaint keeps the source dimensions; nonzero denoise requires multiples of 64.
--prompt-file uses the existing positive/negative fixed/groups preset format.
Saved images are reported as JSON Lines on stdout; diagnostics go to stderr.)", GENESIA_VERSION, genesia::defaults::preset);
            return 0;
        }
        if (option == "--gui") gui = true;
        else if (option == "--preset") {
            name = argument();
            named_preset = true;
        } else if (option == "--prompt-file") {
            prompt_file = argument();
            headless_options = true;
        } else if (option == "--count") {
            number(options.count);
            headless_options = true;
        } else if (option == "--seed") {
            number(options.first_seed.emplace());
            headless_options = true;
        } else if (option == "--source") {
            options.source = argument();
            headless_options = true;
        } else if (option == "--denoise") {
            number(options.denoise.emplace());
            headless_options = true;
        } else throw std::runtime_error{std::format("Unknown option: {}", option)};
    }
    if (options.count <= 0) throw std::invalid_argument{"--count requires a positive integer"};
    if (named_preset && !prompt_file.empty()) throw std::invalid_argument{"Choose either --preset or --prompt-file"};
    if (gui && headless_options) throw std::invalid_argument{"Generation and Repaint options are headless-only"};
    if (options.source.empty() && options.denoise) throw std::invalid_argument{"Repaint options require --source"};
    if (!options.source.empty() && (!options.denoise || !(*options.denoise >= 0 && *options.denoise <= 1))) throw std::invalid_argument{"Repaint requires --denoise in [0,1]"};
#if !defined(GENESIA_HAS_EDITOR)
    if (gui) throw std::runtime_error{"This build does not contain the Editor"};
#endif
    auto catalog = std::make_shared<const genesia::prompt::Catalog>();
    std::optional<genesia::prompt::Preset> preset;
    if (options.source.empty() || named_preset || !prompt_file.empty()) preset = prompt_file.empty() ? genesia::prompt::read_preset(name, *catalog) : genesia::prompt::Preset{prompt_file.stem().string(), genesia::prompt::read_prompt(prompt_file, *catalog)};
#if defined(GENESIA_HAS_EDITOR)
    if (gui) {
        genesia::editor::run(std::move(*preset), std::move(catalog));
        return 0;
    }
#endif
    genesia::headless::run(preset, std::move(catalog), options);
    return 0;
} catch (const std::exception& error) {
    std::println(std::cerr, "Genesia: {}", error.what());
    return 1;
}
