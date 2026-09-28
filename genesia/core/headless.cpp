module;
#include <csignal>
#include <nlohmann/json.hpp>
module genesia.headless;
import genesia.runtime.session;
import genesia.io.files;
import genesia.prompt.presets;
import std;
namespace genesia::headless {
    namespace {
        std::atomic_bool interrupted{};
        void interrupt(const int signal) {
            interrupted.store(true);
            std::signal(signal, interrupt);
        }
    } // namespace
    int run(const std::span<const std::string_view> arguments) {
        if (arguments.empty() || arguments.front() == "--help") {
            std::println(R"(Genesia {}
genesia [--preset NAME]
genesia --headless (--preset NAME | --preset-file FILE | --prompt-file FILE)
                  [--count N] [--seed SEED] [--output DIRECTORY]
                  [--width N] [--height N] [--steps N] [--cfg VALUE]
                  [--lora FILE.safetensors WEIGHT ...] [--lora-start FILE.safetensors FRACTION ...]

Builds with the Editor open it by default. Builds without it default to headless mode.
LoRA files are direct children of assets/loras. Selected filename stems are prepended to the positive prompt.
--lora-start uses the full denoising schedule: 0 = always, 1 = never; default 0.1 (10%).
Prompt files contain positive and negative strings. Presets contain the Editor's free groups and fixed text.
Outputs are independent PNG files in {}. Results and progress are JSON Lines.
--output selects another output directory. --preset-file reads a preset by path.
Ctrl+C stops generation.)",
                GENESIA_VERSION, files::utf8(project::output));
            return 0;
        }
        runtime::Generate operation;
        int count{1};
        std::optional<std::uint64_t> first_seed;
        std::optional<int> width, height, steps;
        std::optional<float> cfg;
        std::optional<std::string> preset_name;
        std::vector<generation::Lora> loras;
        std::map<std::string, float> lora_starts;
        std::filesystem::path prompt_file;
        std::filesystem::path preset_file;
        for (std::size_t i = 0; i < arguments.size(); ++i) {
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
            if (option == "--prompt-file") prompt_file = files::path(argument());
            else if (option == "--preset") preset_name = argument();
            else if (option == "--preset-file") preset_file = files::path(argument());
            else if (option == "--output") operation.output = std::filesystem::absolute(files::path(argument())).lexically_normal();
            else if (option == "--count") number(count);
            else if (option == "--seed") number(first_seed.emplace());
            else if (option == "--width") number(width.emplace());
            else if (option == "--height") number(height.emplace());
            else if (option == "--steps") number(steps.emplace());
            else if (option == "--cfg") number(cfg.emplace());
            else if (option == "--lora") {
                auto& lora = loras.emplace_back();
                lora.file  = argument();
                number(lora.weight);
            } else if (option == "--lora-start") {
                const std::string key{argument()};
                number(lora_starts[key]);
            } else throw std::runtime_error{"Unknown Headless option: " + std::string{option}};
        }
        if (count <= 0) throw std::runtime_error{"Count must be positive"};
        if (int(preset_name.has_value()) + int(!preset_file.empty()) + int(!prompt_file.empty()) != 1) throw std::runtime_error{"Specify one of --preset, --preset-file or --prompt-file"};
        if (!prompt_file.empty()) {
            std::ifstream file{prompt_file};
            file.exceptions(std::ios::badbit | std::ios::failbit);
            const auto json               = nlohmann::json::parse(file);
            operation.parameters.positive = json.at("positive").get<std::string>();
            operation.parameters.negative = json.at("negative").get<std::string>();
        } else {
            const prompt::Catalog catalog;
            const auto preset = preset_name ? prompt::read_preset(*preset_name, catalog) : prompt::read_preset_file(preset_file, catalog);
            operation.parameters.positive = prompt::compose(catalog, preset.prompt.positive);
            operation.parameters.negative = prompt::compose(catalog, preset.prompt.negative);
        }
        if (width) operation.parameters.width = *width;
        if (height) operation.parameters.height = *height;
        if (steps) operation.parameters.steps = *steps;
        if (cfg) operation.parameters.cfg = *cfg;
        for (const auto& [key, start] : lora_starts) {
            const auto selected = std::ranges::find(loras, key, &generation::Lora::file);
            if (selected == loras.end()) throw std::runtime_error{"--lora-start requires --lora for " + key};
            selected->start = start;
        }
        operation.parameters.loras = std::move(loras);
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
#ifdef SIGBREAK
        std::signal(SIGBREAK, interrupt);
#endif
        runtime::Session session;
        operation.count        = count;
        operation.random_seed  = !first_seed;
        operation.seed         = first_seed.value_or(0);
        const auto submitted   = session.submit(std::move(operation));
        const auto& parameters = submitted.request->parameters;
        nlohmann::json started{{"id", submitted.id}, {"kind", "generate"}, {"state", "running"}, {"total", count}, {"width", parameters.width}, {"height", parameters.height}, {"steps", parameters.steps}, {"cfg", parameters.cfg}, {"random_seed", submitted.request->random_seed}};
        if (first_seed) started["seed"] = *first_seed;
        if (!parameters.loras.empty()) started["loras"] = parameters.loras;
        std::println("{}", started.dump());
        std::cout.flush();
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
                    std::println("{}", nlohmann::json{{"id", event.id}, {"kind", "generate"}, {"state", "saved"}, {"completed", completed}, {"total", count}, {"path", files::utf8(event.record.path)}, {"seed", event.record.seed}, {"seconds", {{"sample", event.timing.sample}, {"decode", event.timing.decode}, {"save", event.timing.save}}}}.dump());
                    std::cout.flush();
                }
                if (event.kind != runtime::EventKind::task) continue;
                auto& task = event.task;
                failed |= task.state == runtime::State::failed;
                if (task.state < runtime::State::complete) continue;
                const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - submitted.started).count();
                nlohmann::json json{{"id", task.id}, {"kind", "generate"}, {"state", runtime::states[std::size_t(task.state)]}, {"completed", completed}, {"total", count}, {"seconds", {{"elapsed", elapsed}, {"sample", timing.sample}, {"decode", timing.decode}, {"save", timing.save}}}};
                if (!task.error.empty()) json["error"] = task.error;
                std::println("{}", json.dump());
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
