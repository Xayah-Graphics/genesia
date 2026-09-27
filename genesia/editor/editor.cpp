module;

#include <GLFW/glfw3.h>

module genesia.editor;

import genesia.editor.platform.window;
import genesia.editor.graphics.renderer;
import genesia.editor.workspace;
import genesia.prompt.presets;
import genesia.generation.settings;
import std;

namespace genesia::editor {
    struct Application final {
        WindowPlatform window{"Genesia", {1920, 1080}};
        Renderer renderer{window};
        Workspace ui;
        bool closing{};

        Application(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog);
        void run();
    };

    Application::Application(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog) : ui{std::move(preset), std::move(catalog), window, renderer} {}

    void Application::run() {
        std::uint32_t previous_stage{}, previous_step{};
        bool previous_busy{};
        for (;;) {
            if (window.take_close_request()) {
                closing                  = true;
                ui.continuous_generation = false;
                ui.runtime.web.stop();
                ui.textures.shutdown();
                ui.runtime.session.shutdown();
            }
            bool busy{}, done{true}, pending{ui.textures.pending.load() || ui.runtime.web.enabled.load()};
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
            const bool animating = now < ui.animate_until || (renderer.visible && ui.view.started >= 0) || (ui.prompt_sidebar.amount != float(ui.prompt_sidebar.open));
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
        auto preset  = prompt::read_preset(name, *catalog);
        Application application{std::move(preset), std::move(catalog)};
        application.run();
        return 0;
    }
} // namespace genesia::editor
