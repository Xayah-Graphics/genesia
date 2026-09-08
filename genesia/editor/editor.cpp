module;

#include <GLFW/glfw3.h>

#include "../core/sdxl/control.h"
#include <genesia/cuda.h>

module genesia.editor;

import genesia.editor.platform.window;
import genesia.editor.platform.interop;
import genesia.editor.ui.renderer;
import genesia.editor.session;
import genesia.editor.ui;
import std;

namespace genesia::editor {
    struct Application final {
        WindowPlatform window{"Genesia", {1920, 1080}};
        Renderer renderer{window};
        Interop interop{renderer.device};
        Interop preview_interop{renderer.device};
        Session session;
        UserInterface ui;
        bool closing{};

        Application(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog);
        void run();
    };

    Application::Application(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog) : session{interop, preview_interop}, ui{std::move(preset), std::move(catalog), window, renderer, interop, session} {}

    void Application::run() {
        std::uint32_t previous_stage{}, previous_step{};
        bool previous_busy{};
        for (;;) {
            if (window.take_close_request()) {
                closing = true;
                session.shutdown();
            }
            bool busy, done, pending;
            {
                const std::lock_guard lock{session.mutex};
                busy    = session.active.has_value();
                done    = session.worker_done;
                pending = !session.events.empty() || !session.previews.empty() || ui.gallery.pending.load();
            }
            if (closing && done && !pending) break;
            const auto stage     = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].stage}.load();
            const auto step      = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].completed}.load();
            const double now     = glfwGetTime();
            const bool animating = now < ui.animate_until || (renderer.visible && ui.view.started >= 0) || std::abs(ui.tags_amount - float(ui.tags_open)) > 0.01F || std::abs(ui.gallery_amount - float(ui.gallery_open)) > 0.01F;
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

    void run(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog) {
        Application application{std::move(preset), std::move(catalog)};
        application.run();
    }
} // namespace genesia::editor
