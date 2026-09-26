module;

#include <GLFW/glfw3.h>

module genesia.editor;

import genesia.editor.platform.window;
import genesia.editor.graphics.renderer;
import genesia.editor.workspace;
import genesia.editor.prompt.previews;
import genesia.prompt.library;
import genesia.project;
import genesia.generation.settings;
import genesia.io.files;
import std;

namespace genesia::editor {
    struct Application final {
        WindowPlatform window{"Genesia", {1920, 1080}};
        Renderer renderer{window};
        Workspace ui;
        bool closing{};

        Application(prompts::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, std::shared_ptr<const prompts::Library> library);
        void run();
    };

    Application::Application(prompts::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, std::shared_ptr<const prompts::Library> library) : ui{std::move(preset), std::move(catalog), std::move(library), window, renderer} {}

    void Application::run() {
        std::uint32_t previous_stage{}, previous_step{};
        bool previous_busy{};
        for (;;) {
            if (window.take_close_request()) {
                if (ui.save_model_settings()) {
                    closing                  = true;
                    ui.continuous_generation = false;
                    ui.runtime.web.stop();
                    ui.textures.shutdown();
                    ui.runtime.session.shutdown();
                } else window.redraw = true;
            }
            bool busy{}, done{true}, pending{ui.textures.pending.load() || ui.prompt_panel.images.pending.load() || ui.prompt_panel.images.saving || ui.runtime.web.enabled.load()};
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
            const bool animating = now < ui.animate_until || (renderer.visible && ui.view.started >= 0) || (ui.prompt_sidebar.amount != float(ui.prompt_sidebar.open || ui.prompt_panel.incoming.has_value()));
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
            ui.update_generation();
            ui.update_web();
            renderer.present();
            const double wait = std::min(ui.refresh_at - glfwGetTime(), glfwGetTime() < ui.animate_until || animating ? 1.0 / 120 : busy ? 0.1 : 1.0);
            if (wait > 0) glfwWaitEventsTimeout(wait);
            else glfwPollEvents();
        }
    }

    int run(const std::span<const std::string_view> arguments) {
        std::string name{defaults::preset};
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            const auto option = arguments[i];
            if (i + 1 == arguments.size()) throw std::runtime_error{"Missing value for " + std::string{option}};
            if (option == "--preset") name = arguments[++i];
            else throw std::runtime_error{"Unknown Editor option: " + std::string{option}};
        }
        auto catalog = std::make_shared<const prompt::Catalog>();
        auto library = std::make_shared<const prompts::Library>(std::filesystem::path{project::assets} / "prompts", *catalog);
        auto preset  = prompts::read_preset(*library, name, *catalog);
        previews::clean(*library);
        Application application{std::move(preset), std::move(catalog), std::move(library)};
        application.run();
        return 0;
    }
} // namespace genesia::editor
