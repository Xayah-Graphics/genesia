module;
#include <csignal>
#include <nlohmann/json.hpp>
module genesia.headless;
import genesia.runtime.session;
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
            nlohmann::json json{{"id", task.id}, {"kind", runtime::kinds[std::size_t(task.kind)]}, {"state", runtime::states[std::size_t(task.state)]}};
            if (!task.concept_key.empty()) json["concept"] = task.concept_key;
            if (!task.error.empty()) json["error"] = task.error;
            if (!task.image_sha.empty()) json["image_sha"] = task.image_sha;
            if (!task.model_sha.empty()) json["model_sha"] = task.model_sha;
            std::visit(
                [&]<typename T>(const T& value) {
                    if constexpr (std::same_as<T, runtime::Train>) json["target"] = value.options.steps;
                    else if constexpr (std::same_as<T, runtime::Classify>) json["input"] = files::utf8(value.input);
                    else if constexpr (std::same_as<T, runtime::Fix>) json["category"] = value.category;
                    else if constexpr (std::same_as<T, runtime::Caption>) json["folder"] = value.folder;
                    else if constexpr (std::same_as<T, runtime::Export>) json["output"] = files::utf8(value.output);
                    else if constexpr (std::same_as<T, runtime::Delete> || std::same_as<T, runtime::Normalize>) {
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
                    if constexpr (std::same_as<T, classification::Audit>) {
                        auto& report = json["result"];
                        report       = {{"concept", value.concept_key}, {"model_sha", value.model_sha}, {"fingerprint", value.fingerprint}, {"classes", value.classes}, {"total", value.total}, {"complete", value.complete}, {"rows", nlohmann::json::array()}};
                        for (const auto& row : value.rows) {
                            std::vector<std::string> paths;
                            for (const auto& path : row.sample.paths) paths.push_back(files::utf8(path));
                            report["rows"].push_back({{"sha", row.sample.file.sha}, {"path", files::utf8(row.sample.file.path)}, {"paths", paths}, {"label", row.label}, {"predicted", row.prediction.label}, {"confidence", row.confidence}, {"classes", row.prediction.classes}, {"scores", row.prediction.scores}});
                        }
                    } else if constexpr (std::same_as<T, dataset::MoveResult>) json["result"] = {{"moved", value.paths.size()}};
                    else if constexpr (std::same_as<T, classification::Classification>) json["result"] = {{"input", files::utf8(value.input)}, {"moved", value.movement.paths.size()}, {"classes", value.classes}};
                    else if constexpr (std::same_as<T, dataset::DeleteResult>) {
                        std::vector<std::string> paths;
                        for (const auto& path : value.paths) paths.push_back(files::utf8(path));
                        json["result"] = {{"dataset", value.root}, {"sha", value.sha}, {"deleted", paths.size()}, {"paths", paths}};
                    } else if constexpr (std::same_as<T, dataset::NormalizeResult>) json["result"] = {{"dataset", value.root}, {"files", value.files}, {"linked", value.linked}, {"renamed", value.renamed.size()}};
                    else if constexpr (std::same_as<T, foreground::Result>) json["result"] = {{"path", files::utf8(value.path)}, {"image_sha", value.image.sha}, {"model_sha", value.model_sha}, {"width", value.image.width}, {"height", value.image.height}, {"cached", value.cached}};
                    else if constexpr (std::same_as<T, caption::Exported>) json["result"] = {{"path", files::utf8(value.path)}, {"images", value.images}};
                    else if constexpr (std::same_as<T, runtime::LoraModelResult>) {
                        json["result"] = value.model ? nlohmann::json{{"concept", value.model->id}, {"sha", value.model->sha}, {"name", value.model->name}, {"path", files::utf8(value.model->path)}} : nlohmann::json{};
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
genesia generate --prompt-file FILE [--count N] [--seed SEED]
                 [--width N] [--height N] [--steps N] [--cfg VALUE] [--activate ROOT/CONCEPT ...]
                 [--lora ROOT/CONCEPT WEIGHT ...] [--lora-start ROOT/CONCEPT FRACTION ...]
genesia repaint --source PNG --denoise VALUE [--prompt-file FILE]
                [--count N] [--seed SEED] [--steps N] [--cfg VALUE] [--activate ROOT/CONCEPT ...]
                [--lora ROOT/CONCEPT WEIGHT ...] [--lora-start ROOT/CONCEPT FRACTION ...]
genesia concept ROOT/CONCEPT --type none|classifier|lora
genesia lora ROOT/CONCEPT [--import MODEL.safetensors | --remove]
genesia train ROOT/CONCEPT --steps N [--config FILE] [--restart]
genesia infer ROOT/CONCEPT --input PNG
genesia audit ROOT/CONCEPT [--category NAME] [--refresh]
genesia audit-fix ROOT/CONCEPT --sha SHA --to CATEGORY
genesia classify ROOT/CONCEPT --input DIRECTORY
genesia delete ROOT --sha SHA
genesia normalize ROOT
genesia caption ROOT/CONCEPT [--folder RELATIVE_DIRECTORY] [--tags "tag one, tag two"] [--bypass true|false]
genesia export ROOT/CONCEPT --output NEW_DIRECTORY
genesia mask --input PNG [--output MASK.png]

Training config: physical_batch, effective_batch, head_only_steps, warmup_steps,
head_only_lr, backbone_lr, head_lr, weight_decay, clip_norm, seed,
eval_interval, save_interval, log_interval.
Assign a concept type before training. LoRA concepts manage captions and export for external training.
LoRA import copies one standard SDXL UNet KOHYA_LORA model into the concept and permanently locks its type.
Generate and repaint accept repeated --lora ROOT/CONCEPT WEIGHT options. Trigger words are not added.
--lora-start sets a selected LoRA's start in the full denoising schedule: 0 = always, 1 = never.
It defaults to 0. Repaint uses the same schedule, starting at its existing noise level.
Import replaces the managed model; --remove deletes it without unlocking the type. External files are retained.
LoRA concepts require exactly one path per image SHA, including hard links.
Caption reads or replaces a folder's own tags. Effective captions start with the concept folder name
as the trigger word, followed by tags from the selected folder up to the concept root.
The default caption folder is the concept itself. --tags "" clears its own tags.
--bypass true excludes the folder and its descendants from export; false clears its own bypass.
An ancestor's bypass still applies. Browsing and stored tags are unchanged.
Export copies PNG images, writes full inherited captions and generates foreground -masklabel.png files
into a new directory outside data. Mask uses BiRefNet-general; white is foreground, black is background.
Mask without --output returns the shared cache path. An explicit output must be a new file.
Every non-bypassed directory, including the concept root and empty directories, needs its own tags.
The type locks permanently when the first training record is created.
Data or configuration changes require explicit --restart. Images and type are retained.
Classification moves direct PNG images into DIRECTORY/predicted-class/original-name.
Delete permanently removes all links to the image in ROOT, including its concepts.
Normalize replaces identical PNG copies in ROOT with hard links and numbers each folder's
direct images as 00001.png, 00002.png, ... by modification time. Dot directories are skipped.
All generation outputs are saved into the project data/raw directory.
Dataset PNG images do not require a Genesia generation record.
Prompt files contain final positive and negative strings. Headless does not load Editor presets.
Repaint requires a version 2 Genesia generation record, including when --prompt-file is supplied.
Results and progress are JSON Lines. Ctrl+C stops at a safe task boundary.)",
                GENESIA_VERSION);
            return 0;
        }
        const auto command = arguments.front();
        const bool repaint = command == "repaint", generating = command == "generate" || repaint;
        runtime::Request request;
        std::size_t begin = 1;
        if (command == "mask") request.operation = runtime::Mask{};
        else if (!generating) {
            if (arguments.size() < 2) throw std::runtime_error{command == "delete" || command == "normalize" ? "Specify ROOT" : "Specify ROOT/CONCEPT"};
            const std::string key{arguments[1]};
            begin = 2;
            if (command == "train") request.operation = runtime::Train{{.concept_key = key}};
            else if (command == "infer") request.operation = runtime::Infer{.concept_key = key};
            else if (command == "audit") request.operation = runtime::Audit{key};
            else if (command == "audit-fix") request.operation = runtime::Fix{key};
            else if (command == "classify") request.operation = runtime::Classify{key};
            else if (command == "concept") request.operation = runtime::Assign{key};
            else if (command == "delete") request.operation = runtime::Delete{key};
            else if (command == "normalize") request.operation = runtime::Normalize{key};
            else if (command == "caption") request.operation = runtime::Caption{key};
            else if (command == "export") request.operation = runtime::Export{key};
            else if (command == "lora") request.operation = runtime::LoraModel{key};
            else throw std::runtime_error{"Unknown command: " + std::string{command}};
        }
        int count{1};
        std::optional<std::uint64_t> first_seed;
        std::optional<int> width, height, steps;
        std::optional<float> cfg, denoise;
        std::vector<std::string> activated;
        std::vector<generation::Lora> loras;
        std::map<std::string, float> lora_starts;
        std::filesystem::path source, prompt_file, config_file;
        std::string category;
        bool steps_set{}, type_set{};
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
            if (option == "--prompt-file" && generating) prompt_file = files::path(argument());
            else if (option == "--count" && generating) number(count);
            else if (option == "--seed" && generating) number(first_seed.emplace());
            else if (option == "--width" && command == "generate") number(width.emplace());
            else if (option == "--height" && command == "generate") number(height.emplace());
            else if (option == "--steps" && generating) number(steps.emplace());
            else if (option == "--cfg" && generating) number(cfg.emplace());
            else if (option == "--lora" && generating) {
                auto& lora       = loras.emplace_back();
                lora.concept_key = argument();
                number(lora.weight);
            } else if (option == "--lora-start" && generating) {
                const std::string key{argument()};
                number(lora_starts[key]);
            } else if (option == "--source" && repaint) source = files::path(argument());
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
            else if (option == "--input" && command == "mask") std::get<runtime::Mask>(request.operation).input = files::path(argument());
            else if (option == "--output" && command == "mask") std::get<runtime::Mask>(request.operation).output = files::path(argument());
            else if (option == "--input" && command == "infer") std::get<runtime::Infer>(request.operation).input = files::path(argument());
            else if (option == "--input" && command == "classify") std::get<runtime::Classify>(request.operation).input = files::path(argument());
            else if (option == "--category" && command == "audit") category = argument();
            else if (option == "--refresh" && command == "audit") std::get<runtime::Audit>(request.operation).refresh = true;
            else if (option == "--sha" && command == "audit-fix") std::get<runtime::Fix>(request.operation).sha = argument();
            else if (option == "--sha" && command == "delete") std::get<runtime::Delete>(request.operation).sha = argument();
            else if (option == "--to" && command == "audit-fix") std::get<runtime::Fix>(request.operation).category = argument();
            else if (option == "--folder" && command == "caption") std::get<runtime::Caption>(request.operation).folder = files::utf8(files::path(argument()).lexically_normal());
            else if (option == "--tags" && command == "caption") std::get<runtime::Caption>(request.operation).tags = caption::parse(argument());
            else if (option == "--bypass" && command == "caption") {
                const auto value = argument();
                if (value != "true" && value != "false") throw std::runtime_error{"--bypass requires true or false"};
                std::get<runtime::Caption>(request.operation).bypass = value == "true";
            } else if (option == "--output" && command == "export") std::get<runtime::Export>(request.operation).output = files::path(argument());
            else if (option == "--import" && command == "lora") std::get<runtime::LoraModel>(request.operation).input = files::path(argument());
            else if (option == "--remove" && command == "lora") std::get<runtime::LoraModel>(request.operation).remove = true;
            else throw std::runtime_error{"Unknown option for this command: " + std::string{option}};
        }
        if (count <= 0) throw std::runtime_error{"Count must be positive"};
        if (command == "generate" && prompt_file.empty()) throw std::runtime_error{"Generate requires --prompt-file"};
        if (repaint && (source.empty() || !denoise || *denoise < 0 || *denoise > 1)) throw std::runtime_error{"Repaint requires --source and --denoise in [0,1]"};
        if (command == "train" && !steps_set) throw std::runtime_error{"Training requires --steps"};
        if (command == "concept" && !type_set) throw std::runtime_error{"Specify --type none, classifier or lora"};
        if (command == "export" && std::get<runtime::Export>(request.operation).output.empty()) throw std::runtime_error{"Specify --output NEW_DIRECTORY"};
        if (command == "lora" && std::get<runtime::LoraModel>(request.operation).remove && !std::get<runtime::LoraModel>(request.operation).input.empty()) throw std::runtime_error{"Choose --import or --remove"};
        if (generating) {
            auto& operation   = std::get<runtime::Generate>(request.operation);
            if (repaint) {
                source           = std::filesystem::absolute(source).lexically_normal();
                const auto record = read_record(source);
                operation.parameters = record.parameters;
                dataset::Index index;
                operation.source             = runtime::RepaintSource{index.identify(source, std::array{record.parameters.width, record.parameters.height}).sha, source};
                operation.parameters.denoise = *denoise;
            }
            if (!prompt_file.empty()) {
                std::ifstream file{prompt_file};
                file.exceptions(std::ios::badbit | std::ios::failbit);
                const auto json = nlohmann::json::parse(file);
                operation.parameters.positive = json.at("positive").get<std::string>();
                operation.parameters.negative = json.at("negative").get<std::string>();
            }
            if (width) operation.parameters.width = *width;
            if (height) operation.parameters.height = *height;
            if (steps) operation.parameters.steps = *steps;
            if (cfg) operation.parameters.cfg = *cfg;
            for (const auto& [key, start] : lora_starts) {
                const auto selected = std::ranges::find(loras, key, &generation::Lora::concept_key);
                if (selected == loras.end()) throw std::runtime_error{"--lora-start requires --lora for " + key};
                selected->start = start;
            }
            operation.parameters.loras = std::move(loras);
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
        const auto submitted = session.submit(std::move(request));
        if (generating) {
            const auto& operation  = std::get<runtime::Generate>(submitted.request->operation);
            const auto& parameters = operation.parameters;
            nlohmann::json json{{"id", submitted.id}, {"kind", "generate"}, {"state", "running"}, {"total", operation.count}, {"width", parameters.width}, {"height", parameters.height}, {"steps", parameters.steps}, {"cfg", parameters.cfg}, {"random_seed", operation.random_seed}};
            if (!operation.random_seed) json["seed"] = operation.seed;
            if (!parameters.loras.empty()) json["loras"] = parameters.loras;
            if (operation.source) {
                json["source"]  = files::utf8(operation.source->path.lexically_relative(project::directory));
                json["denoise"] = parameters.denoise;
            }
            std::println("{}", json.dump());
            std::cout.flush();
        }
        std::size_t completed{};
        runtime::GenerationTiming timing;
        bool failed{}, shutting_down{};
        for (;;) {
            if (interrupted.load() && !shutting_down) {
                session.shutdown();
                shutting_down = true;
            }
            auto delivery = session.drain();
            for (auto& event : delivery.events) {
                if (event.kind == runtime::EventKind::saved) {
                    ++completed;
                    timing.sample += event.timing.sample;
                    timing.decode += event.timing.decode;
                    timing.save += event.timing.save;
                    std::println("{}", nlohmann::json{{"id", event.id}, {"kind", "generate"}, {"state", "saved"}, {"completed", completed}, {"total", count}, {"path", files::utf8(event.record.path)}, {"seed", event.record.seed}, {"image_sha", event.file->sha}, {"seconds", {{"sample", event.timing.sample}, {"decode", event.timing.decode}, {"save", event.timing.save}}}}.dump());
                    std::cout.flush();
                }
                if (event.kind != runtime::EventKind::task) continue;
                auto& task = event.task;
                failed |= task.state == runtime::State::failed;
                if (task.kind == runtime::Kind::generate) {
                    if (task.state < runtime::State::complete) continue;
                    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - submitted.started).count();
                    nlohmann::json json{{"id", task.id}, {"kind", "generate"}, {"state", runtime::states[std::size_t(task.state)]}, {"completed", completed}, {"total", count}, {"seconds", {{"elapsed", elapsed}, {"sample", timing.sample}, {"decode", timing.decode}, {"save", timing.save}}}};
                    if (!task.error.empty()) json["error"] = task.error;
                    std::println("{}", json.dump());
                    std::cout.flush();
                    continue;
                }
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
