module;
#include <csignal>
#include <nlohmann/json.hpp>
module edit.headless;
import edit.session;
import edit.preset;
import tools.files;
import std;
namespace edit::headless {
    namespace {
        std::atomic_bool interrupted{};
        void interrupt(int signal) {
            interrupted = true;
            std::signal(signal, interrupt);
        }
    } // namespace
    int run(const std::span<const std::string_view> arguments) {
        if (arguments.empty() || arguments.front() == "--help") {
            std::println(R"(Edit {}
edit
edit --headless INPUT --prompt TEXT [--reference FILE ...]
edit --headless INPUT --prompt-file FILE [--reference FILE ...]
edit --headless INPUT --preset NAME

All headless modes accept [--output DIRECTORY] [--suffix TEXT].

Editor builds open a small drag-and-drop window by default.
Headless builds accept arguments without --headless.
INPUT is a PNG image or a directory containing PNG images (not recursive).
Each input image is <image1>. Up to nine --reference PNG files are fixed
as <image2> through <image10>, in argument order, for the entire batch.
--preset loads prompt and fixed references from assets/edit/preset/NAME.json.
It cannot be combined with --prompt, --prompt-file or --reference.
Outputs go into INPUT/fix, or into the image's parent/fix, with original names.
--output selects another directory. --suffix is inserted before the extension.
Existing results are replaced only after a complete output is ready. Originals remain.
Original ComfyUI outputs are also retained in the configured output directory.
One Qwen Image 2.1 edit per image, with a random seed and the same positive prompt.
Each batch starts a Python worker, loads its models once, and exits after completion.
Runtime paths are configured in assets/edit/runtime.json. No ComfyUI server is needed.
Progress and results are JSON Lines. Ctrl+C stops this batch.)",
                EDIT_VERSION);
            return 0;
        }
        Request request{.input = tools::files::path(arguments.front())};
        bool prompt_set{};
        std::optional<std::string> preset_name;
        for (std::size_t i = 1; i < arguments.size(); ++i) {
            const auto option = arguments[i];
            if (option != "--prompt" && option != "--prompt-file" && option != "--reference" && option != "--preset" && option != "--output" && option != "--suffix") throw std::runtime_error{"Unknown option: " + std::string{option}};
            if (++i == arguments.size()) throw std::runtime_error{"Missing value for " + std::string{option}};
            if (option == "--output") request.output = tools::files::path(arguments[i]);
            else if (option == "--suffix") request.suffix = arguments[i];
            else if (option == "--preset") {
                if (preset_name) throw std::runtime_error{"Specify only one --preset"};
                preset_name = arguments[i];
            } else if (option == "--reference") {
                if (request.references.size() == 9) throw std::runtime_error{"Edit accepts at most nine fixed reference images"};
                request.references.push_back(tools::files::path(arguments[i]));
            } else {
                if (prompt_set) throw std::runtime_error{"Specify only one of --prompt or --prompt-file"};
                request.prompt = option == "--prompt" ? std::string{arguments[i]} : tools::files::read_text(tools::files::path(arguments[i]));
                prompt_set     = true;
            }
        }
        if (preset_name) {
            if (prompt_set || !request.references.empty()) throw std::runtime_error{"--preset cannot be combined with --prompt, --prompt-file or --reference"};
            auto preset        = read_preset(std::move(*preset_name));
            request.prompt     = std::move(preset.prompt);
            request.references = std::move(preset.references);
        } else if (!prompt_set) throw std::runtime_error{"Specify --prompt, --prompt-file or --preset"};
        interrupted = false;
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
        runtime::Session session;
        session.submit(std::move(request));
        bool stopping{};
        for (;;) {
            if (interrupted && !stopping) {
                session.cancel();
                stopping = true;
            }
            const auto delivery = session.drain();
            for (const auto& progress : delivery.progress) {
                const nlohmann::json event{{"state", "running"}, {"progress", {{"stage", stages[std::size_t(progress.stage)]}, {"completed", progress.completed}, {"total", progress.total}, {"file", tools::files::utf8(progress.file)}}}};
                std::println("{}", event.dump());
            }
            std::cout.flush();
            if (!delivery.state.busy) {
                const auto& result = delivery.state.result;
                nlohmann::json event{{"state", runtime::states[std::size_t(delivery.state.state)]}, {"result", {{"output", tools::files::utf8(result.output)}, {"completed", result.completed}, {"total", result.total}}}};
                if (!result.error.empty()) event["error"] = result.error;
                std::println("{}", event.dump());
                return !result.error.empty() ? 1 : result.stopped ? 130 : 0;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
        }
    }
} // namespace edit::headless
