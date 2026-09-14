module;

#include <GLFW/glfw3.h>

module genesia.editor;

import genesia.editor.platform.window;
import genesia.editor.graphics.renderer;
import genesia.editor.workspace;
import genesia.prompt.preset;
import genesia.io.files;
import std;

namespace genesia::editor {
    struct Application final {
        WindowPlatform window{"Genesia", {1920, 1080}};
        Renderer renderer{window};
        Workspace ui;
        bool closing{};

        Application(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, std::string dataset);
        void run();
    };

    Application::Application(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, std::string dataset) : ui{std::move(preset), std::move(catalog), window, renderer, std::move(dataset)} {}

    void Application::run() {
        std::uint32_t previous_stage{}, previous_step{};
        bool previous_busy{};
        for (;;) {
            if (window.take_close_request()) {
                closing = true;
                if (ui.runtime) ui.runtime->session.shutdown();
            }
            bool busy{}, done{true}, pending{ui.dataset_index.pending.load() || ui.textures.pending.load()};
            std::uint32_t stage{}, step{};
            if (ui.runtime) {
                const auto state = ui.runtime->session.snapshot();
                busy             = state.active.has_value();
                done             = state.finished;
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
        auto preset  = prompt::read_preset(name, *catalog);
        Application application{std::move(preset), std::move(catalog), std::move(dataset)};
        application.run();
        return 0;
    }
} // namespace genesia::editor
