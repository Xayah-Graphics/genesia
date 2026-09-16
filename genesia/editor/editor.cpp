module;

#include <GLFW/glfw3.h>

module genesia.editor;

import genesia.editor.platform.window;
import genesia.editor.graphics.renderer;
import genesia.editor.workspace;
import genesia.prompt.library;
import genesia.project;
import genesia.generation.defaults;
import genesia.io.files;
import std;

namespace genesia::editor {
    struct Application final {
        WindowPlatform window{"Genesia", {1920, 1080}};
        Renderer renderer{window};
        Workspace ui;
        bool closing{};

        Application(prompts::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, std::shared_ptr<const prompts::Library> library, std::string dataset);
        void run();
    };

    Application::Application(prompts::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, std::shared_ptr<const prompts::Library> library, std::string dataset) : ui{std::move(preset), std::move(catalog), std::move(library), window, renderer, std::move(dataset)} {}

    void Application::run() {
        std::uint32_t previous_stage{}, previous_step{};
        bool previous_busy{};
        for (;;) {
            if (window.take_close_request()) {
                if (ui.save_model_settings()) {
                    closing = true;
                    ui.runtime.session.shutdown();
                } else window.redraw = true;
            }
            bool busy{}, done{true}, pending{ui.textures.pending.load() || ui.prompt_panel.images.pending.load() || ui.prompt_panel.images.saving};
            std::uint32_t stage{}, step{};
            {
                const auto state = ui.runtime.session.snapshot();
                busy             = state.active.has_value();
                done             = state.finished;
                if (closing && done && !state.error.empty()) throw std::runtime_error{state.error};
                pending |= state.pending;
                stage = static_cast<std::uint32_t>(state.generation.stage);
                step  = state.generation.completed;
            }
            if (closing && done && !pending) break;
            const double now     = glfwGetTime();
            const bool animating = now < ui.animate_until || (renderer.visible && ui.view.started >= 0) || std::ranges::any_of(std::array{&ui.dataset_sidebar, &ui.prompt_sidebar}, [](const auto* panel) { return panel->amount != float(panel->open); });
            if (!std::exchange(window.redraw, false) && !pending && !animating && now < ui.refresh_at && stage == previous_stage && step == previous_step && busy == previous_busy) {
                glfwWaitEventsTimeout(std::min(busy ? 0.1 : 1.0, std::max(0.0, ui.refresh_at - now)));
                continue;
            }
            previous_stage = stage;
            previous_step  = step;
            previous_busy  = busy;
            if (!renderer.begin()) {
                glfwPollEvents();
                continue;
            }
            ui.receive();
            ui.draw();
            renderer.present();
            const double wait = std::min(ui.refresh_at - glfwGetTime(), glfwGetTime() < ui.animate_until || animating ? 1.0 / 120 : busy ? 0.1 : 1.0);
            if (wait > 0) glfwWaitEventsTimeout(wait);
            else glfwPollEvents();
        }
    }

    int run(const std::span<const std::string_view> arguments) {
        std::string name{defaults::preset}, dataset;
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            const auto option = arguments[i];
            if (i + 1 == arguments.size()) throw std::runtime_error{"Missing value for " + std::string{option}};
            if (option == "--preset") name = arguments[++i];
            else if (option == "--dataset") dataset = arguments[++i];
            else throw std::runtime_error{"Unknown Editor option: " + std::string{option}};
        }
        if (!dataset.empty()) {
            const auto path  = files::path(dataset);
            const auto count = std::distance(path.begin(), path.end());
            if (path.is_absolute() || count < 1 || count > 2 || std::ranges::any_of(path, [](const auto& part) { return files::utf8(part).starts_with('.'); })) throw std::runtime_error{"Dataset must be ROOT or ROOT/CONCEPT"};
            dataset = files::utf8(path);
        }
        auto catalog = std::make_shared<const prompt::Catalog>();
        auto library = std::make_shared<const prompts::Library>(std::filesystem::path{project::assets} / "prompts", *catalog);
        auto preset  = prompts::read_preset(*library, name, *catalog);
        Application application{std::move(preset), std::move(catalog), std::move(library), std::move(dataset)};
        application.run();
        return 0;
    }
} // namespace genesia::editor
