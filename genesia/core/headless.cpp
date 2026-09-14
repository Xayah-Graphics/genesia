module;
#include <csignal>
#include <nlohmann/json.hpp>
module genesia.headless;
import genesia.runtime.session;
import genesia.prompt.preset;
import genesia.io.files;
import std;
namespace genesia::headless {
    namespace {
        std::atomic_bool interrupted{};
        void interrupt(const int signal) {
            interrupted.store(true);
            std::signal(signal, interrupt);
        }
        nlohmann::json event_json(const runtime::TaskStatus& task) {
            nlohmann::json json{{"id", task.id}, {"kind", runtime::kinds[std::size_t(task.kind)]}, {"concept", task.concept_key}, {"state", runtime::states[std::size_t(task.state)]}};
            if (!task.error.empty()) json["error"] = task.error;
            if (!task.image_sha.empty()) json["image_sha"] = task.image_sha;
            if (!task.model_sha.empty()) json["model_sha"] = task.model_sha;
            std::visit(
                [&]<typename T>(const T& value) {
                    if constexpr (std::same_as<T, runtime::Generate>) {
                        json["source"] = value.source ? files::utf8(value.source->path.lexically_relative(project::directory)) : "";
                        json["seed"]   = value.seed;
                        json["width"]  = value.parameters.width;
                        json["height"] = value.parameters.height;
                    } else if constexpr (std::same_as<T, runtime::Train>) json["target"] = value.options.steps;
                    else if constexpr (std::same_as<T, runtime::Classify>) json["input"] = files::utf8(value.input);
                    else if constexpr (std::same_as<T, runtime::Fix>) json["category"] = value.category;
                    else if constexpr (std::same_as<T, runtime::Delete>) {
                        json.erase("concept");
                        json["dataset"] = value.root;
                    }
                },
                task.request->operation);
            std::visit(
                [&]<typename T>(const T& value) {
                    if constexpr (std::same_as<T, runtime::BatchProgress>) json["progress"] = {{"stage", runtime::stages[std::size_t(value.stage)]}, {"completed", value.completed}, {"total", value.total}};
                    else if constexpr (std::same_as<T, training::Step>) {
                        json["progress"]          = value;
                        json["progress"]["stage"] = "training";
                    } else if constexpr (std::same_as<T, training::Metrics>) json["progress"] = {{"stage", "evaluation"}, {"metrics", value}};
                },
                task.progress.value);
            std::visit(
                [&]<typename T>(const T& value) {
                    if constexpr (std::same_as<T, runtime::Generated>) json["result"] = {{"path", files::utf8(value.path)}, {"seed", value.seed}};
                    else if constexpr (std::same_as<T, classification::Audit>) {
                        auto& report = json["result"];
                        report       = {{"concept", value.concept_key}, {"model_sha", value.model_sha}, {"fingerprint", value.fingerprint}, {"classes", value.classes}, {"total", value.total}, {"complete", value.complete}, {"rows", nlohmann::json::array()}};
                        for (const auto& row : value.rows) {
                            std::vector<std::string> paths;
                            for (const auto& path : row.sample.paths) paths.push_back(files::utf8(path));
                            report["rows"].push_back({{"sha", row.sample.file.sha}, {"path", files::utf8(row.sample.file.path)}, {"paths", paths}, {"label", row.label}, {"predicted", row.prediction.label}, {"confidence", row.confidence}, {"classes", row.prediction.classes}, {"scores", row.prediction.scores}});
                        }
                    } else if constexpr (std::same_as<T, dataset::MoveResult>) json["result"] = {{"moved", value.moved}, {"restored", value.restored}};
                    else if constexpr (std::same_as<T, classification::Classification>) json["result"] = {{"input", files::utf8(value.input)}, {"moved", value.movement.moved}, {"classes", value.classes}};
                    else if constexpr (std::same_as<T, dataset::DeleteResult>) {
                        std::vector<std::string> paths;
                        for (const auto& path : value.paths) paths.push_back(files::utf8(path));
                        json["result"] = {{"dataset", value.root}, {"sha", value.sha}, {"deleted", paths.size()}, {"paths", paths}};
                    } else if constexpr (!std::same_as<T, std::monostate>) json["result"] = value;
                },
                task.result.value);
            return json;
        }
    } // namespace
    int run(const std::span<const std::string_view> arguments) {
        if (arguments.empty() || arguments.front() == "--help") {
            std::println(R"(Genesia {}
genesia --gui [--preset NAME] [--dataset ROOT[/CONCEPT]]
genesia generate [--preset NAME | --prompt-file FILE] [--count N] [--seed SEED]
                 [--width N] [--height N] [--steps N] [--cfg VALUE] [--activate ROOT/CONCEPT ...]
genesia repaint --source PNG --denoise VALUE [--preset NAME | --prompt-file FILE]
                [--count N] [--seed SEED] [--steps N] [--cfg VALUE] [--activate ROOT/CONCEPT ...]
genesia concept ROOT/CONCEPT --type none|classifier|lora
genesia train ROOT/CONCEPT --steps N [--config FILE] [--restart]
genesia infer ROOT/CONCEPT --input PNG
genesia audit ROOT/CONCEPT [--category NAME] [--refresh]
genesia audit-fix ROOT/CONCEPT --sha SHA --to CATEGORY
genesia audit-undo ROOT/CONCEPT
genesia classify ROOT/CONCEPT --input DIRECTORY
genesia delete ROOT --sha SHA

Training config: physical_batch, effective_batch, head_only_steps, warmup_steps,
head_only_lr, backbone_lr, head_lr, weight_decay, clip_norm, seed,
eval_interval, save_interval, log_interval.
Assign a concept type before training. LoRA can be assigned but cannot train yet.
The type locks permanently when the first training record is created.
Data or configuration changes require explicit --restart. Images and type are retained.
Classification moves direct PNG images into DIRECTORY/predicted-class/original-name.
Delete permanently removes all links to the image in ROOT, including its concepts.
All generation outputs are saved into the project data/raw directory.
Results and progress are JSON Lines. Ctrl+C stops at a safe task boundary.)",
                GENESIA_VERSION);
            return 0;
        }
        const auto command = arguments.front();
        const bool repaint = command == "repaint", generating = command == "generate" || repaint;
        runtime::Request request;
        std::size_t begin = 1;
        if (!generating) {
            if (arguments.size() < 2) throw std::runtime_error{command == "delete" ? "Specify ROOT" : "Specify ROOT/CONCEPT"};
            const std::string key{arguments[1]};
            begin = 2;
            if (command == "train") request.operation = runtime::Train{{.concept_key = key}};
            else if (command == "infer") request.operation = runtime::Infer{.concept_key = key};
            else if (command == "audit") request.operation = runtime::Audit{key};
            else if (command == "audit-fix") request.operation = runtime::Fix{key};
            else if (command == "audit-undo") request.operation = runtime::Undo{key};
            else if (command == "classify") request.operation = runtime::Classify{key};
            else if (command == "concept") request.operation = runtime::Assign{key};
            else if (command == "delete") request.operation = runtime::Delete{key};
            else throw std::runtime_error{"Unknown command: " + std::string{command}};
        }
        int count{1};
        std::optional<std::uint64_t> first_seed;
        std::optional<int> width, height, steps;
        std::optional<float> cfg, denoise;
        std::vector<std::string> activated;
        std::filesystem::path source, prompt_file, config_file;
        std::string name{defaults::preset}, category;
        bool named_preset{}, steps_set{}, type_set{};
        for (std::size_t i = begin; i < arguments.size(); ++i) {
            const auto option   = arguments[i];
            const auto argument = [&]() {
                if (i + 1 == arguments.size()) throw std::runtime_error{"Missing value for " + std::string{option}};
                return arguments[++i];
            };
            const auto number = [&](auto& destination) {
                const auto text   = argument();
                const auto parsed = std::from_chars(text.data(), text.data() + text.size(), destination);
                if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) throw std::runtime_error{"Invalid value for " + std::string{option}};
            };
            if (option == "--preset" && generating) {
                name         = argument();
                named_preset = true;
            } else if (option == "--prompt-file" && generating) prompt_file = files::path(argument());
            else if (option == "--count" && generating) number(count);
            else if (option == "--seed" && generating) number(first_seed.emplace());
            else if (option == "--width" && command == "generate") number(width.emplace());
            else if (option == "--height" && command == "generate") number(height.emplace());
            else if (option == "--steps" && generating) number(steps.emplace());
            else if (option == "--cfg" && generating) number(cfg.emplace());
            else if (option == "--source" && repaint) source = files::path(argument());
            else if (option == "--denoise" && repaint) number(denoise.emplace());
            else if (option == "--activate" && generating) {
                activated.emplace_back(argument());
                while (i + 1 < arguments.size() && !arguments[i + 1].starts_with("--")) activated.emplace_back(arguments[++i]);
            } else if (option == "--type" && command == "concept") {
                std::get<runtime::Assign>(request.operation).type = dataset::parse_concept_type(argument());
                type_set                                          = true;
            } else if (option == "--steps" && command == "train") {
                number(std::get<runtime::Train>(request.operation).options.steps);
                steps_set = true;
            } else if (option == "--config" && command == "train") config_file = files::path(argument());
            else if (option == "--restart" && command == "train") std::get<runtime::Train>(request.operation).options.restart = true;
            else if (option == "--input" && command == "infer") std::get<runtime::Infer>(request.operation).input = files::path(argument());
            else if (option == "--input" && command == "classify") std::get<runtime::Classify>(request.operation).input = files::path(argument());
            else if (option == "--category" && command == "audit") category = argument();
            else if (option == "--refresh" && command == "audit") std::get<runtime::Audit>(request.operation).refresh = true;
            else if (option == "--sha" && command == "audit-fix") std::get<runtime::Fix>(request.operation).sha = argument();
            else if (option == "--sha" && command == "delete") std::get<runtime::Delete>(request.operation).sha = argument();
            else if (option == "--to" && command == "audit-fix") std::get<runtime::Fix>(request.operation).category = argument();
            else throw std::runtime_error{"Unknown option for this command: " + std::string{option}};
        }
        if (count <= 0) throw std::runtime_error{"Count must be positive"};
        if (named_preset && !prompt_file.empty()) throw std::runtime_error{"Choose a preset or a prompt file"};
        if (repaint && (source.empty() || !denoise || *denoise < 0 || *denoise > 1)) throw std::runtime_error{"Repaint requires --source and --denoise in [0,1]"};
        if (command == "train" && !steps_set) throw std::runtime_error{"Training requires --steps"};
        if (command == "concept" && !type_set) throw std::runtime_error{"Specify --type none, classifier or lora"};
        if (generating) {
            auto& operation   = std::get<runtime::Generate>(request.operation);
            operation.catalog = std::make_shared<const prompt::Catalog>();
            if (repaint) {
                source               = std::filesystem::absolute(source).lexically_normal();
                const auto record    = read_record(source);
                auto resolved        = prompt::resolve(record.prompt, operation.catalog);
                operation.parameters = record.parameters;
                operation.prompt     = std::move(resolved.prompt);
                operation.catalog    = std::move(resolved.catalog);
                dataset::Index index;
                operation.source             = runtime::RepaintSource{index.identify(source).sha, source};
                operation.parameters.denoise = *denoise;
            }
            if (!repaint || named_preset || !prompt_file.empty()) operation.prompt = prompt_file.empty() ? prompt::read_preset(name, *operation.catalog).prompt : prompt::read_prompt(prompt_file, *operation.catalog);
            operation.parameters.positive = prompt::compose(*operation.catalog, operation.prompt.positive);
            operation.parameters.negative = prompt::compose(*operation.catalog, operation.prompt.negative);
            if (width) operation.parameters.width = *width;
            if (height) operation.parameters.height = *height;
            if (steps) operation.parameters.steps = *steps;
            if (cfg) operation.parameters.cfg = *cfg;
        } else if (command == "train" && !config_file.empty()) {
            auto& options       = std::get<runtime::Train>(request.operation).options;
            const auto assigned = dataset::read_concept(options.concept_key);
            const auto saved    = training::read_state(assigned.path);
            options.config      = training::read_config(config_file, !options.restart && saved ? saved->config : training::Config{});
        }
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
#ifdef SIGBREAK
        std::signal(SIGBREAK, interrupt);
#endif
        runtime::Session session;
        session.activate(activated);
        if (generating) {
            auto& operation       = std::get<runtime::Generate>(request.operation);
            operation.count       = count;
            operation.random_seed = !first_seed;
            operation.seed        = first_seed.value_or(0);
        }
        session.submit(std::move(request));
        bool failed{}, shutting_down{};
        for (;;) {
            if (interrupted.load() && !shutting_down) {
                session.shutdown();
                shutting_down = true;
            }
            auto delivery = session.drain();
            for (auto& event : delivery.events) {
                if (event.kind == runtime::EventKind::saved) {
                    std::println("{}", nlohmann::json{{"id", event.id}, {"kind", "generate"}, {"state", "saved"}, {"path", files::utf8(event.record.path)}, {"seed", event.record.seed}, {"image_sha", event.file->sha}}.dump());
                    std::cout.flush();
                }
                if (event.kind != runtime::EventKind::task) continue;
                auto& task = event.task;
                failed |= task.state == runtime::State::failed;
                if (task.kind == runtime::Kind::audit && task.state == runtime::State::complete && !category.empty()) {
                    auto& report = std::get<classification::Audit>(task.result.value);
                    if (!std::ranges::contains(report.classes, category)) throw std::runtime_error{"Unknown audit category: " + category};
                    std::erase_if(report.rows, [&](const classification::AuditRow& row) { return row.label != category; });
                    report.total = report.rows.size();
                }
                std::println("{}", event_json(task).dump());
                std::cout.flush();
            }
            const auto state = session.snapshot();
            if (!state.error.empty()) throw std::runtime_error{state.error};
            if (state.idle && !state.pending && !shutting_down) {
                session.shutdown();
                shutting_down = true;
            }
            if (state.finished && !state.pending) break;
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
        return interrupted.load() ? 130 : failed ? 1 : 0;
    }
} // namespace genesia::headless
