module;

#include <GLFW/glfw3.h>
#include "../core/sdxl/control.h"
#include <genesia/cuda.h>

module genesia.editor;

import genesia.editor.platform.window;
import genesia.editor.ui.renderer;
import genesia.editor.ui;
import std;

namespace genesia::editor {
    struct Application final {
        WindowPlatform window{"Genesia", {1920, 1080}};
        Renderer renderer{window};
        UserInterface ui;
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
            bool busy{}, done{true}, pending{ui.library.pending.load()};
            std::uint32_t stage{}, step{};
            if (ui.runtime) {
                auto& session = ui.runtime->session;
                const std::lock_guard lock{session.mutex};
                busy = session.active.has_value();
                done = session.worker_done;
                pending |= !session.events.empty() || !session.previews.empty();
                stage = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].stage}.load();
                step  = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].completed}.load();
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

    void run(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, std::string dataset) {
        Application application{std::move(preset), std::move(catalog), std::move(dataset)};
        application.run();
    }
} // namespace genesia::editor
