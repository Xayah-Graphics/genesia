module;
#include <csignal>
#include <nlohmann/json.hpp>
module qwen.headless;
import qwen.session;
import tools.files;
import std;
namespace qwen::headless {
    namespace {
        std::atomic_bool interrupted{};
        void interrupt(int signal) {
            interrupted = true;
            std::signal(signal, interrupt);
        }
    } // namespace
    int run(const std::span<const std::string_view> arguments) {
        if (arguments.empty() || arguments.front() == "--help") {
            std::println(R"(Qwen {}
qwen
qwen --headless INPUT --prompt TEXT [--choice RESULT ...]

INPUT is a PNG image or a directory containing PNG images (not recursive).
The default candidates are YES and NO. A single image is only analyzed.
Folder inputs move each PNG into a subfolder named after its result.
Existing destination files are never overwritten. Results contain the final paths.
Results are printed as JSON Lines. LM Studio: {}, model: {}.)",
                QWEN_VERSION, server_url, model);
            return 0;
        }
        Request request{.input = tools::files::path(arguments.front()), .choices = {"YES", "NO"}};
        bool prompt_set{}, choices_set{};
        for (std::size_t i = 1; i < arguments.size(); ++i) {
            const auto option = arguments[i];
            if (option != "--prompt" && option != "--prompt-file" && option != "--choice") throw std::runtime_error{"Unknown option: " + std::string{option}};
            if (++i == arguments.size()) throw std::runtime_error{"Missing value for " + std::string{option}};
            if (option == "--prompt") {
                if (prompt_set) throw std::runtime_error{"Specify only one prompt"};
                request.prompt = std::string{arguments[i]};
                prompt_set     = true;
            } else if (option == "--prompt-file") {
                if (prompt_set) throw std::runtime_error{"Specify only one prompt"};
                request.prompt = tools::files::read_text(tools::files::path(arguments[i]));
                prompt_set     = true;
            } else {
                if (!choices_set) request.choices.clear();
                choices_set = true;
                request.choices.emplace_back(arguments[i]);
            }
        }
        if (!prompt_set || request.prompt.find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error{"Specify a non-empty --prompt or --prompt-file"};
        if (request.choices.size() < 2) throw std::runtime_error{"Specify at least two --choice values"};
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
            for (const auto& progress : delivery.progress)
                if (!progress.result.empty()) std::println("{}", nlohmann::json{{"image", tools::files::utf8(progress.file)}, {"result", progress.result}}.dump());
            std::cout.flush();
            if (!delivery.state.busy) {
                if (!delivery.state.result.error.empty()) std::println("{}", nlohmann::json{{"state", "failed"}, {"error", delivery.state.result.error}}.dump());
                else std::println("{}", nlohmann::json{{"state", runtime::states[std::size_t(delivery.state.state)]}, {"completed", delivery.state.result.items.size()}, {"total", delivery.state.result.total}}.dump());
                return !delivery.state.result.error.empty() ? 1 : delivery.state.result.stopped ? 130 : 0;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
        }
    }
} // namespace qwen::headless
