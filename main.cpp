#include <nlohmann/json.hpp>
import genesia.headless;
import genesia.dataset;
import genesia.files;
#if defined(GENESIA_HAS_EDITOR)
import genesia.editor;
#endif
import std;

int main(const int argc, char** argv) try {
    if (argc == 1 || std::string_view{argv[1]} == "--help") {
        std::println(R"(Genesia {}
genesia --gui [--preset NAME] [--dataset ROOT[/CONCEPT]]
genesia generate [--preset NAME | --prompt-file FILE] [--count N] [--seed SEED]
                 [--width N] [--height N] [--steps N] [--cfg VALUE]
                 [--activate ROOT/CONCEPT ...]
genesia repaint --source PNG --denoise VALUE [--preset NAME | --prompt-file FILE]
                [--count N] [--seed SEED] [--steps N] [--cfg VALUE]
                [--activate ROOT/CONCEPT ...]
genesia concept ROOT/CONCEPT --type none|classifier|lora
genesia train ROOT/CONCEPT --steps N [--config FILE] [--restart]
genesia infer ROOT/CONCEPT --input PNG
genesia audit ROOT/CONCEPT [--category NAME] [--refresh]
genesia audit-fix ROOT/CONCEPT --sha SHA --to CATEGORY
genesia audit-undo ROOT/CONCEPT
genesia classify ROOT/CONCEPT --input DIRECTORY

Training config: physical_batch, effective_batch, head_only_steps, warmup_steps,
head_only_lr, backbone_lr, head_lr, weight_decay, clip_norm, seed,
eval_interval, save_interval, log_interval.
Assign a concept type before training. LoRA can be assigned but cannot train yet.
The type locks permanently when the first training record is created.
Data or configuration changes require explicit --restart, which deletes classifier
training state and uses the initial pretrained model. Images and type are retained.
Classification moves direct PNG images into DIRECTORY/predicted-class/original-name.
All generation outputs are saved into the project data/raw directory.
Results and progress are JSON Lines. Ctrl+C stops at a safe task boundary.)",
            GENESIA_VERSION);
        return 0;
    }
    const std::string_view command{argv[1]};
    const bool gui        = command == "--gui";
    const bool repaint    = command == "repaint";
    const bool generating = command == "generate" || repaint;
    const bool assigning  = command == "concept";
    genesia::headless::Options options;
    const std::map<std::string_view, genesia::work::Kind> commands{{"train", genesia::work::Kind::train}, {"infer", genesia::work::Kind::infer}, {"audit", genesia::work::Kind::audit}, {"audit-fix", genesia::work::Kind::fix}, {"audit-undo", genesia::work::Kind::undo}, {"classify", genesia::work::Kind::classify}};
    int begin = 2;
    if (!gui && !generating) {
        const auto found = commands.find(command);
        if (!assigning && found == commands.end()) throw std::runtime_error{"Unknown command: " + std::string(command)};
        if (!assigning) options.request.kind = found->second;
        if (argc < 3) throw std::runtime_error{"Specify ROOT/CONCEPT"};
        options.request.concept_key          = argv[2];
        options.request.training.concept_key = argv[2];
        begin                                = 3;
    }
    bool named_preset{}, steps_set{};
    std::optional<genesia::dataset::ConceptType> assigned_type;
    std::string name{genesia::defaults::preset}, dataset;
    std::filesystem::path prompt_file;
    for (int i = begin; i < argc; ++i) {
        const std::string_view option{argv[i]};
        const auto argument = [&]() -> std::string_view {
            if (i + 1 == argc) throw std::invalid_argument{"Missing value for " + std::string(option)};
            return argv[++i];
        };
        const auto number = [&](auto& destination) {
            const auto value  = argument();
            const auto result = std::from_chars(value.data(), value.data() + value.size(), destination);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) throw std::invalid_argument{"Invalid value for " + std::string(option)};
        };
        const bool training = options.request.kind == genesia::work::Kind::train;
        if (option == "--dataset" && gui) dataset = argument();
        else if (option == "--preset" && (generating || gui)) {
            name         = argument();
            named_preset = true;
        } else if (option == "--prompt-file" && generating) prompt_file = genesia::files::path(argument());
        else if (option == "--count" && generating) number(options.count);
        else if (option == "--seed" && generating) number(options.first_seed.emplace());
        else if (option == "--width" && command == "generate") number(options.width.emplace());
        else if (option == "--height" && command == "generate") number(options.height.emplace());
        else if (option == "--steps" && generating) number(options.steps.emplace());
        else if (option == "--cfg" && generating) number(options.cfg.emplace());
        else if (option == "--source" && repaint) options.source = genesia::files::path(argument());
        else if (option == "--denoise" && repaint) number(options.denoise.emplace());
        else if (option == "--activate" && generating) {
            options.activated.emplace_back(argument());
            while (i + 1 < argc && !std::string_view{argv[i + 1]}.starts_with("--")) options.activated.emplace_back(argv[++i]);
        } else if (option == "--type" && assigning) assigned_type = genesia::dataset::parse_concept_type(argument());
        else if (option == "--steps" && training) {
            number(options.request.training.steps);
            steps_set = true;
        } else if (option == "--config" && training) options.request.training.overrides = genesia::files::read_json(genesia::files::path(argument()));
        else if (option == "--restart" && training) options.request.training.restart = true;
        else if (option == "--input" && (command == "infer" || command == "classify")) options.request.input = genesia::files::path(argument());
        else if (option == "--category" && command == "audit") options.request.category = argument();
        else if (option == "--refresh" && command == "audit") options.request.refresh = true;
        else if (option == "--sha" && command == "audit-fix") options.request.sha = argument();
        else if (option == "--to" && command == "audit-fix") options.request.category = argument();
        else throw std::runtime_error{"Unknown option for this command: " + std::string(option)};
    }
    if (options.count <= 0) throw std::runtime_error{"Count must be positive"};
    if (named_preset && !prompt_file.empty()) throw std::runtime_error{"Choose a preset or a prompt file"};
    if (repaint && (options.source.empty() || !options.denoise || !(*options.denoise >= 0 && *options.denoise <= 1))) throw std::runtime_error{"Repaint requires --source and --denoise in [0,1]"};
    if (command == "train" && !steps_set) throw std::runtime_error{"Training requires --steps"};
    if (assigning) {
        if (!assigned_type) throw std::runtime_error{"Specify --type none, classifier or lora"};
        const auto assigned = genesia::dataset::assign_type(options.request.concept_key, *assigned_type);
        std::println("{}", nlohmann::json{{"concept", assigned.key}, {"state", "complete"}, {"result", assigned}}.dump());
        return 0;
    }
    if ((command == "infer" || command == "classify") && options.request.input.empty()) throw std::runtime_error{"Specify --input"};
    if (command == "audit-fix" && (options.request.sha.empty() || options.request.category.empty())) throw std::runtime_error{"Specify --sha and --to"};
    if (!dataset.empty()) {
        const auto path  = genesia::files::path(dataset);
        const auto count = std::distance(path.begin(), path.end());
        if (path.is_absolute() || count < 1 || count > 2 || std::ranges::any_of(path, [](const auto& part) { return genesia::files::utf8(part).starts_with('.'); })) throw std::runtime_error{"Dataset must be ROOT or ROOT/CONCEPT"};
        dataset = genesia::files::utf8(path);
    }
    std::shared_ptr<const genesia::prompt::Catalog> catalog;
    std::optional<genesia::prompt::Preset> preset;
    if (gui || generating) {
        catalog = std::make_shared<const genesia::prompt::Catalog>();
        if (!repaint || named_preset || !prompt_file.empty()) preset = prompt_file.empty() ? genesia::prompt::read_preset(name, *catalog) : genesia::prompt::Preset{genesia::files::utf8(prompt_file.stem()), genesia::prompt::read_prompt(prompt_file, *catalog)};
    }
    if (gui) {
#if defined(GENESIA_HAS_EDITOR)
        genesia::editor::run(std::move(*preset), std::move(catalog), std::move(dataset));
        return 0;
#else
        throw std::runtime_error{"This build does not contain the Editor"};
#endif
    }
    return genesia::headless::run(preset, std::move(catalog), options);
} catch (const std::exception& error) {
    std::println(std::cerr, "Genesia: {}", error.what());
    return 1;
}
