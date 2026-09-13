#include <csignal>

#include <nlohmann/json.hpp>

import std;
import classifier.training;
import classifier.audit;
import classifier.classify;

namespace {

    static_assert(std::atomic_bool::is_always_lock_free);

    std::atomic_bool interrupted = false;

    void interrupt_work(int signal) {
        std::signal(signal, interrupt_work);
        interrupted.store(true, std::memory_order_relaxed);
    }

    enum class Mode { train, audit, classify };

    std::filesystem::path utf8_path(const std::string& value) {
        return std::filesystem::path(std::u8string(value.begin(), value.end()));
    }

} // namespace

int main(int argc, char** argv) {
    try {
        classifier::TrainingOptions options;

        Mode mode          = Mode::train;
        bool mode_selected = false;
        bool steps_set     = false;

        std::filesystem::path model_path;
        std::filesystem::path classify_input;

        const std::map<std::string, std::string> integers{{"--batch", "physical_batch"}, {"--effective-batch", "effective_batch"}, {"--freeze-steps", "head_only_steps"}, {"--warmup-steps", "warmup_steps"}, {"--eval-interval", "eval_interval"}, {"--save-interval", "save_interval"}, {"--log-interval", "log_interval"}};

        const std::map<std::string, std::string> decimals{{"--backbone-lr", "backbone_lr"}, {"--head-lr", "head_lr"}, {"--frozen-head-lr", "head_only_lr"}, {"--weight-decay", "weight_decay"}, {"--clip-norm", "clip_norm"}};

        const auto select_mode = [&](Mode selected, std::string_view option) {
            if (mode_selected && mode != selected) {
                throw std::runtime_error(std::format("{} conflicts with another operating mode", option));
            }

            mode          = selected;
            mode_selected = true;
        };

        for (int i = 1; i < argc; ++i) {
            std::string option = argv[i];

            if (option == "--help") {
                std::println("Training:\n"
                             "  classifier-train --dataset <YES/NO folders root> "
                             "--steps <cumulative target>\n"
                             "  --restart  Start a new task and archive existing results\n"
                             "  --batch 4 --effective-batch 64 "
                             "--freeze-steps 25 --warmup-steps 10\n"
                             "  --backbone-lr 3e-5 --head-lr 3e-4 "
                             "--frozen-head-lr 1e-3\n"
                             "  --weight-decay 0.01 --clip-norm 1 --seed 42\n"
                             "  --eval-interval 25 --save-interval 25 "
                             "--log-interval 5\n"
                             "\n"
                             "Label audit (no training):\n"
                             "  classifier-train --dataset <directory> --audit "
                             "[--model <safetensors>]\n"
                             "  Default model: <dataset>/model.safetensors\n"
                             "  Report: <dataset>/.classifier/audit.html\n"
                             "\n"
                             "Automatic classification (no training):\n"
                             "  classifier-train --dataset <training root> "
                             "--classify <directory> [--model <safetensors>]\n"
                             "  Default model: <dataset>/model.safetensors\n"
                             "  Only images directly inside <directory> are classified.\n"
                             "  Subdirectories are not searched.\n"
                             "  Images are moved in place to <directory>/<predicted class>/.\n"
                             "  Each image is renamed to its complete class-score list,\n"
                             "  ordered from highest to lowest score with two decimals.\n"
                             "\n"
                             "Classification first scores the complete input set. "
                             "Files are moved only after all images were scored successfully.\n"
                             "The model's stored YES threshold is used unchanged.\n"
                             "\n"
                             "Ctrl+C during training saves after the current complete update.\n"
                             "Ctrl+C during classification scoring stops before any files are moved.");

                return 0;
            }

            if (option == "--restart") {
                options.restart = true;
                continue;
            }

            if (option == "--audit") {
                select_mode(Mode::audit, option);
                continue;
            }

            if (i + 1 == argc) {
                throw std::runtime_error("Missing value for " + option);
            }

            std::string value = argv[++i];

            if (option == "--dataset") {
                options.dataset = utf8_path(value);
            } else if (option == "--model") {
                model_path = utf8_path(value);
            } else if (option == "--classify") {
                select_mode(Mode::classify, option);

                if (!classify_input.empty()) {
                    throw std::runtime_error("--classify was specified more than once");
                }

                classify_input = utf8_path(value);
            } else if (option == "--steps") {
                options.steps = std::stoi(value);
                steps_set     = true;
            } else if (option == "--seed") {
                options.overrides["seed"] = std::stoull(value);
            } else if (integers.contains(option)) {
                options.overrides[integers.at(option)] = std::stoi(value);
            } else if (decimals.contains(option)) {
                options.overrides[decimals.at(option)] = std::stof(value);
            } else {
                throw std::runtime_error("Unknown option: " + option);
            }
        }

        if (options.dataset.empty()) {
            throw std::runtime_error("Specify --dataset; use --help for options");
        }

        if (mode == Mode::train) {
            if (!model_path.empty()) {
                throw std::runtime_error("--model requires --audit or --classify");
            }
        } else {
            if (steps_set || options.restart || !options.overrides.empty()) {

                throw std::runtime_error("Training options cannot be combined "
                                         "with --audit or --classify");
            }
        }

        if (mode == Mode::classify && classify_input.empty()) {

            throw std::runtime_error("--classify requires a directory");
        }

        std::signal(SIGINT, interrupt_work);
        std::signal(SIGTERM, interrupt_work);

#ifdef SIGBREAK
        std::signal(SIGBREAK, interrupt_work);
#endif

        switch (mode) {
        case Mode::train: classifier::train(options, interrupted); break;

        case Mode::audit: classifier::audit(options.dataset, model_path, interrupted); break;

        case Mode::classify:
            {
                classifier::ClassifyOptions classify_options;
                classify_options.dataset = options.dataset;
                classify_options.input   = classify_input;
                classify_options.model   = model_path;

                classifier::classify(classify_options, interrupted);

                break;
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::println(stderr, "{}", e.what());

        return 1;
    }
}
