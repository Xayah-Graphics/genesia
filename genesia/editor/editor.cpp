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
        WindowPlatform window{"Genesia", {1440, 960}};
        Renderer renderer{window};
        Interop interop{renderer.device};
        Interop preview_interop{renderer.device};
        Session session;
        UserInterface ui;
        bool closing{};
        std::vector<double> frame_times;

        Application(Configuration configuration, std::filesystem::path path);
        void run();
    };

    Application::Application(Configuration configuration, std::filesystem::path path) : session{configuration, interop, preview_interop}, ui{std::move(configuration), std::move(path), window, renderer, interop, session} {}

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
                pending = !session.events.empty() || !session.previews.empty();
            }
            if (closing && done && !pending) break;
            const auto stage     = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].stage}.load();
            const auto step      = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].completed}.load();
            const double now     = glfwGetTime();
            const bool animating = now < ui.animate_until || std::abs(ui.composer_amount - float(ui.composer_open)) > 0.01F || std::abs(ui.history_amount - float(ui.history_open)) > 0.01F;
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
            if (busy && glfwGetTime() < ui.animate_until) frame_times.push_back(renderer.last_frame_seconds);
            const double wait = std::min(ui.refresh_at - glfwGetTime(), glfwGetTime() < ui.animate_until || animating ? 1.0 / 120 : busy ? 0.1 : 1.0);
            if (wait > 0) glfwWaitEventsTimeout(wait);
            else glfwPollEvents();
        }
        if (!frame_times.empty()) {
            std::ranges::sort(frame_times);
            std::println("EDITOR interactive frame median={:.2f}ms p95={:.2f}ms", frame_times[frame_times.size() / 2] * 1000, frame_times[std::min(frame_times.size() - 1, frame_times.size() * 95 / 100)] * 1000);
        }
        if (!ui.display_times.empty()) {
            std::ranges::sort(ui.display_times);
            std::println("EDITOR result ready-to-submit median={:.2f}ms", ui.display_times[ui.display_times.size() / 2] * 1000);
        }
    }

    void run(Configuration configuration, const std::filesystem::path& configuration_path) {
        Application application{std::move(configuration), configuration_path};
        application.run();
    }
} // namespace genesia::editor
