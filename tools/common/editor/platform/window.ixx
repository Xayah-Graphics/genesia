module;

#include <Windows.h>
#include <GLFW/glfw3.h>
#include <ole2.h>

export module tools.editor.platform.window;

import std;
import vulkan;

namespace tools::editor {
    export struct WindowPlatform {
        explicit WindowPlatform(std::string_view application_name, vk::Extent2D initial_extent);
        ~WindowPlatform();

        WindowPlatform(const WindowPlatform&)            = delete;
        WindowPlatform(WindowPlatform&&)                 = delete;
        WindowPlatform& operator=(const WindowPlatform&) = delete;
        WindowPlatform& operator=(WindowPlatform&&)      = delete;

        [[nodiscard]] bool take_close_request() noexcept;
        void move();
        void resize(vk::Extent2D extent);

        GLFWwindow* window{};
        HWND native_window{};
        vk::Extent2D content_extent{}, extent_limit{};
        bool redraw{true}, drag_requested{};
        std::vector<std::filesystem::path> dragged, dropped;
        std::string drop_error;

    private:
        struct GlfwLifetime {
            GlfwLifetime();
            ~GlfwLifetime();

            GlfwLifetime(const GlfwLifetime&)            = delete;
            GlfwLifetime(GlfwLifetime&&)                 = delete;
            GlfwLifetime& operator=(const GlfwLifetime&) = delete;
            GlfwLifetime& operator=(GlfwLifetime&&)      = delete;
        };
        struct OleLifetime {
            OleLifetime();
            ~OleLifetime();
        };
        struct DropTarget final : IDropTarget {
            explicit DropTarget(WindowPlatform& window);
            HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** object) override;
            ULONG STDMETHODCALLTYPE AddRef() override;
            ULONG STDMETHODCALLTYPE Release() override;
            HRESULT STDMETHODCALLTYPE DragEnter(IDataObject* data, DWORD keys, POINTL point, DWORD* effect) override;
            HRESULT STDMETHODCALLTYPE DragOver(DWORD keys, POINTL point, DWORD* effect) override;
            HRESULT STDMETHODCALLTYPE DragLeave() override;
            HRESULT STDMETHODCALLTYPE Drop(IDataObject* data, DWORD keys, POINTL point, DWORD* effect) override;

            WindowPlatform& window;
            std::atomic<ULONG> references{1};
        };

        static LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam);
        struct {
            GlfwLifetime glfw{};
            OleLifetime ole{};
            std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> glfw_window{nullptr, glfwDestroyWindow};
            WNDPROC original_window_proc{};
            bool close_requested{};
        } state;
        DropTarget drop_target{*this};
    };
} // namespace tools::editor
