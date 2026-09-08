module;

#include <Windows.h>

#include <GLFW/glfw3.h>

export module genesia.editor.platform.window;

import std;
import vulkan;

namespace genesia::editor {
    export struct WindowPlatform {
        explicit WindowPlatform(std::string_view application_name, vk::Extent2D initial_extent);
        ~WindowPlatform();

        WindowPlatform(const WindowPlatform&)            = delete;
        WindowPlatform(WindowPlatform&&)                 = delete;
        WindowPlatform& operator=(const WindowPlatform&) = delete;
        WindowPlatform& operator=(WindowPlatform&&)      = delete;

        void request_close() noexcept;
        [[nodiscard]] bool take_close_request() noexcept;
        void toggle_fullscreen();
        void prepare_hand_cursors(float scale);

        GLFWwindow* window{};
        HWND native_window{};
        std::array<float, 4> drag_region{};
        bool redraw{true};
        std::array<GLFWcursor*, 2> hand_cursors{};

    private:
        struct GlfwLifetime {
            GlfwLifetime();
            ~GlfwLifetime();

            GlfwLifetime(const GlfwLifetime&)            = delete;
            GlfwLifetime(GlfwLifetime&&)                 = delete;
            GlfwLifetime& operator=(const GlfwLifetime&) = delete;
            GlfwLifetime& operator=(GlfwLifetime&&)      = delete;
        };

        static LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam);
        struct {
            GlfwLifetime glfw{};
            std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> glfw_window{nullptr, glfwDestroyWindow};
            WNDPROC original_window_proc{};
            WINDOWPLACEMENT windowed_placement{sizeof(WINDOWPLACEMENT)};
            LONG_PTR windowed_style{};
            bool fullscreen{};
            bool close_requested{};
        } state;
    };
} // namespace genesia::editor
