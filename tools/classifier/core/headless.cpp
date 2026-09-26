module;
#include <csignal>
#include <nlohmann/json.hpp>
module classifier.headless;
import classifier.session;
import std;
namespace classifier::headless {
    namespace {
        std::atomic_bool interrupted{};
        void interrupt(int signal) {
            interrupted = true;
            std::signal(signal, interrupt);
        }
    } // namespace
    int run(const std::span<const std::string_view> arguments) {
        if (arguments.empty() || arguments.front() == "--help") {
            std::println(R"(Classifier {}
classifier
classifier --headless train DIRECTORY [--steps N] [--config FILE] [--restart]
classifier --headless audit DIRECTORY [--category NAME] [--refresh]
classifier --headless audit-fix DIRECTORY --sha SHA --to CATEGORY
classifier --headless infer DIRECTORY --input PNG [--refresh]
classifier --headless classify DIRECTORY --input INPUT_DIRECTORY

Editor builds open a small drag-and-drop window by default.
Headless builds accept commands without --headless.
Directories may be anywhere on disk. Existing .genesia classifiers can resume.
Training defaults to 400 cumulative steps. Data or parameter changes require --restart.
Inputs: PNG, 1024x1024, 1024x1536 or 1536x1024, without resizing.
Classify moves direct PNG files into INPUT_DIRECTORY/predicted-class/original-name.
Audit-fix moves every path of the selected sample to the specified category.
Progress and results are JSON Lines. Ctrl+C requests a safe stop.)",
                CLASSIFIER_VERSION);
            return 0;
        }
        if (arguments.size() < 2) throw std::runtime_error{"Specify a classifier directory"};
        const auto command = arguments.front();
        const auto root    = tools::files::path(arguments[1]);
        runtime::Request request;
        if (command == "train") request.operation = runtime::Train{{.root = root}};
        else if (command == "audit") request.operation = runtime::Audit{root};
        else if (command == "audit-fix") request.operation = runtime::Fix{root};
        else if (command == "infer") request.operation = runtime::Infer{root};
        else if (command == "classify") request.operation = runtime::Classify{root};
        else throw std::runtime_error{"Unknown command: " + std::string{command}};
        std::string category;
        for (std::size_t i = 2; i < arguments.size(); ++i) {
            const auto option = arguments[i];
            const auto value  = [&]() {
                if (i + 1 == arguments.size()) throw std::runtime_error{"Missing value for " + std::string{option}};
                return arguments[++i];
            };
            if (command == "train" && option == "--steps") {
                const auto text   = value();
                const auto parsed = std::from_chars(text.data(), text.data() + text.size(), std::get<runtime::Train>(request.operation).options.steps);
                if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) throw std::runtime_error{"Invalid step count"};
            } else if (command == "train" && option == "--config") std::get<runtime::Train>(request.operation).options.config = training::read_config(tools::files::path(value()));
            else if (command == "train" && option == "--restart") std::get<runtime::Train>(request.operation).options.restart = true;
            else if (command == "audit" && option == "--category") category = value();
            else if (command == "audit" && option == "--refresh") std::get<runtime::Audit>(request.operation).refresh = true;
            else if (command == "audit-fix" && option == "--sha") std::get<runtime::Fix>(request.operation).sha = value();
            else if (command == "audit-fix" && option == "--to") std::get<runtime::Fix>(request.operation).category = value();
            else if (command == "infer" && option == "--input") std::get<runtime::Infer>(request.operation).input = tools::files::path(value());
            else if (command == "infer" && option == "--refresh") std::get<runtime::Infer>(request.operation).refresh = true;
            else if (command == "classify" && option == "--input") std::get<runtime::Classify>(request.operation).input = tools::files::path(value());
            else throw std::runtime_error{"Unknown option for this command: " + std::string{option}};
        }
        if (command == "infer" && std::get<runtime::Infer>(request.operation).input.empty()) throw std::runtime_error{"Specify --input PNG"};
        if (command == "classify" && std::get<runtime::Classify>(request.operation).input.empty()) throw std::runtime_error{"Specify --input INPUT_DIRECTORY"};
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
            auto delivery = session.drain();
            for (const auto& progress : delivery.progress) {
                nlohmann::json event{{"kind", command}, {"state", "running"}};
                std::visit(
                    [&]<typename T>(const T& value) {
                        if constexpr (std::same_as<T, runtime::BatchProgress>) event["progress"] = {{"stage", runtime::stages[std::size_t(value.stage)]}, {"completed", value.completed}, {"total", value.total}, {"file", value.file}};
                        else if constexpr (std::same_as<T, training::Step>) event["progress"] = {{"stage", "training"}, {"metric", value}};
                        else if constexpr (std::same_as<T, training::Metrics>) event["progress"] = {{"stage", "evaluation"}, {"metric", value}};
                    },
                    progress.value);
                std::println("{}", event.dump());
            }
            std::cout.flush();
            const auto& state = delivery.state;
            if (!state.busy) {
                nlohmann::json event{{"kind", command}, {"state", runtime::states[std::size_t(state.state)]}};
                if (!state.error.empty()) event["error"] = state.error;
                if (command == "train" && state.training && state.training->state) event["result"] = *state.training->state;
                if (command == "audit" && state.audit) {
                    const auto& value = *state.audit;
                    if (!category.empty() && !std::ranges::contains(value.classes, category)) throw std::runtime_error{"Unknown audit category: " + category};
                    auto rows = nlohmann::json::array();
                    for (const auto& row : value.rows) {
                        if (!category.empty() && category != row.label) continue;
                        std::vector<std::string> paths;
                        for (const auto& path : row.sample.paths) paths.push_back(tools::files::utf8(path));
                        rows.push_back({{"sha", row.sample.file.sha}, {"path", tools::files::utf8(row.sample.file.path)}, {"paths", paths}, {"label", row.label}, {"confidence", row.confidence}, {"prediction", row.prediction}});
                    }
                    event["result"] = {{"model_sha", value.model_sha}, {"fingerprint", value.fingerprint}, {"classes", value.classes}, {"total", rows.size()}, {"rows", std::move(rows)}};
                }
                if (state.result)
                    std::visit(
                        [&]<typename T>(const T& value) {
                            if constexpr (std::same_as<T, runtime::Inferred>) event["result"] = {{"path", tools::files::utf8(value.path)}, {"prediction", value.prediction}};
                            else if constexpr (std::same_as<T, dataset::MoveResult>) event["result"] = {{"moved", value.paths.size()}};
                            else if constexpr (std::same_as<T, classification::Classification>) event["result"] = {{"input", tools::files::utf8(value.input)}, {"moved", value.movement.paths.size()}, {"classes", value.classes}};
                        },
                        state.result->value);
                std::println("{}", event.dump());
                return interrupted ? 130 : state.state == runtime::State::failed ? 1 : 0;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
        }
    }
} // namespace classifier::headless
