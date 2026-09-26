module;
#include <csignal>
#include <nlohmann/json.hpp>
module foreground.headless;
import foreground.session;
import tools.files;
import std;
namespace foreground::headless {
    namespace {
        std::atomic_bool interrupted{};
        void interrupt(int signal) {
            interrupted = true;
            std::signal(signal, interrupt);
        }
    } // namespace
    int run(const std::span<const std::string_view> arguments) {
        if (arguments.empty() || arguments.front() == "--help") {
            std::println(R"(Foreground {}
foreground
foreground --headless DIRECTORY

Editor builds open a small drag-and-drop window by default.
Headless builds accept arguments without --headless.
Recursively processes PNG images, excluding existing -masklabel.png masks.
Each image gets a same-size grayscale mask beside it: NAME-masklabel.png.
White is foreground; black is background. Existing masks are overwritten.
Progress and results are JSON Lines. Ctrl+C stops after the current image.)",
                FOREGROUND_VERSION);
            return 0;
        }
        if (arguments.size() != 1) throw std::runtime_error{"Specify one input directory"};
        Request request{.input = tools::files::path(arguments.front())};
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
                std::cout.flush();
                return !result.error.empty() ? 1 : result.stopped ? 130 : 0;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
        }
    }
} // namespace foreground::headless
