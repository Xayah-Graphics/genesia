module;
#include <csignal>
#include <nlohmann/json.hpp>
module edit.headless;
import edit.session;
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
edit --headless INPUT --prompt TEXT
edit --headless INPUT --prompt-file FILE

Editor builds open a small drag-and-drop window by default.
Headless builds accept arguments without --headless.
INPUT is a PNG image or a directory containing PNG images (not recursive).
Outputs go into INPUT/fix, or into the image's parent/fix, with original names.
Existing results are replaced only after a complete output is ready. Originals remain.
One Qwen Image 2.1 edit per image, with a random seed and the same positive prompt.
Local ComfyUI: {}, with --input-directory and --output-directory.
Progress and results are JSON Lines. Ctrl+C stops this batch.)",
                EDIT_VERSION, server_url);
            return 0;
        }
        Request request{.input = tools::files::path(arguments.front())};
        bool prompt_set{};
        for (std::size_t i = 1; i < arguments.size(); ++i) {
            const auto option = arguments[i];
            if (option != "--prompt" && option != "--prompt-file") throw std::runtime_error{"Unknown option: " + std::string{option}};
            if (prompt_set) throw std::runtime_error{"Specify only one of --prompt or --prompt-file"};
            if (++i == arguments.size()) throw std::runtime_error{"Missing value for " + std::string{option}};
            request.prompt = option == "--prompt" ? std::string{arguments[i]} : tools::files::read_text(tools::files::path(arguments[i]));
            prompt_set     = true;
        }
        if (!prompt_set) throw std::runtime_error{"Specify --prompt or --prompt-file"};
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
