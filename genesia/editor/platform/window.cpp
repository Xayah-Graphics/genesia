module;

#include <Windows.h>

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <dwmapi.h>
#include <windowsx.h>

module genesia.editor.platform.window;

import std;
import vulkan;

namespace genesia::editor {
    WindowPlatform::WindowPlatform(const std::string_view application_name, const vk::Extent2D initial_extent) {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        this->state.glfw_window.reset(glfwCreateWindow(static_cast<int>(initial_extent.width), static_cast<int>(initial_extent.height), std::string{application_name}.c_str(), nullptr, nullptr));
        if (!this->state.glfw_window) throw std::runtime_error("Genesia window creation failed");
        this->window        = this->state.glfw_window.get();
        this->native_window = glfwGetWin32Window(this->window);
        glfwSetWindowUserPointer(this->window, this);
        glfwSetWindowCloseCallback(this->window, [](GLFWwindow* window) {
            WindowPlatform& platform       = *static_cast<WindowPlatform*>(glfwGetWindowUserPointer(window));
            platform.state.close_requested = true;
            glfwSetWindowShouldClose(window, GLFW_FALSE);
        });

        SetPropW(this->native_window, L"GenesiaWindow", this);
        this->state.original_window_proc = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(this->native_window, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&WindowPlatform::window_proc)));
        constexpr LONG_PTR style         = WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU;
        SetWindowLongPtrW(this->native_window, GWL_STYLE, style);
        constexpr BOOL dark_mode = TRUE;
        DwmSetWindowAttribute(this->native_window, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark_mode, sizeof(dark_mode));
        constexpr DWM_WINDOW_CORNER_PREFERENCE corners = DWMWCP_ROUND;
        DwmSetWindowAttribute(this->native_window, DWMWA_WINDOW_CORNER_PREFERENCE, &corners, sizeof(corners));
        MONITORINFO monitor_info{sizeof(MONITORINFO)};
        GetMonitorInfoW(MonitorFromWindow(this->native_window, MONITOR_DEFAULTTONEAREST), &monitor_info);
        RECT bounds{};
        GetWindowRect(this->native_window, &bounds);
        const auto& area = monitor_info.rcWork;
        const auto x = area.left + ((area.right - area.left) - (bounds.right - bounds.left)) / 2;
        const auto y = area.top + ((area.bottom - area.top) - (bounds.bottom - bounds.top)) / 2;
        SetWindowPos(this->native_window, nullptr, x, y, 0, 0, SWP_FRAMECHANGED | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
        glfwShowWindow(this->window);
    }

    WindowPlatform::~WindowPlatform() {
        SetWindowLongPtrW(this->native_window, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(this->state.original_window_proc));
        RemovePropW(this->native_window, L"GenesiaWindow");
    }

    void WindowPlatform::request_close() noexcept {
        this->state.close_requested = true;
    }

    bool WindowPlatform::take_close_request() noexcept {
        return std::exchange(this->state.close_requested, false);
    }

    void WindowPlatform::toggle_fullscreen() {
        if (!state.fullscreen) {
            GetWindowPlacement(native_window, &state.windowed_placement);
            state.windowed_style = GetWindowLongPtrW(native_window, GWL_STYLE);
            MONITORINFO monitor_info{sizeof(MONITORINFO)};
            GetMonitorInfoW(MonitorFromWindow(native_window, MONITOR_DEFAULTTONEAREST), &monitor_info);
            state.fullscreen = true;
            SetWindowLongPtrW(native_window, GWL_STYLE, state.windowed_style & ~static_cast<LONG_PTR>(WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MAXIMIZE));
            constexpr DWM_WINDOW_CORNER_PREFERENCE corners = DWMWCP_DONOTROUND;
            DwmSetWindowAttribute(native_window, DWMWA_WINDOW_CORNER_PREFERENCE, &corners, sizeof(corners));
            const auto& area = monitor_info.rcMonitor;
            SetWindowPos(native_window, HWND_TOP, area.left, area.top, area.right - area.left, area.bottom - area.top, SWP_FRAMECHANGED | SWP_NOOWNERZORDER);
        } else {
            state.fullscreen = false;
            SetWindowLongPtrW(native_window, GWL_STYLE, state.windowed_style);
            SetWindowPlacement(native_window, &state.windowed_placement);
            constexpr DWM_WINDOW_CORNER_PREFERENCE corners = DWMWCP_ROUND;
            DwmSetWindowAttribute(native_window, DWMWA_WINDOW_CORNER_PREFERENCE, &corners, sizeof(corners));
            SetWindowPos(native_window, nullptr, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
        }
        redraw = true;
    }

    WindowPlatform::GlfwLifetime::GlfwLifetime() {
        if (glfwInit() != GLFW_TRUE) throw std::runtime_error("GLFW initialization failed");
    }

    WindowPlatform::GlfwLifetime::~GlfwLifetime() {
        glfwTerminate();
    }

    LRESULT CALLBACK WindowPlatform::window_proc(HWND window, const UINT message, const WPARAM wparam, const LPARAM lparam) {
        WindowPlatform* platform = static_cast<WindowPlatform*>(GetPropW(window, L"GenesiaWindow"));
        if (platform == nullptr) return DefWindowProcW(window, message, wparam, lparam);
        if ((message >= WM_MOUSEFIRST && message <= WM_MOUSELAST) || (message >= WM_KEYFIRST && message <= WM_KEYLAST) || message == WM_SIZE || message == WM_DPICHANGED || message == WM_PAINT || message == WM_SETFOCUS || message == WM_KILLFOCUS) platform->redraw = true;
        switch (message) {
        case WM_NCCALCSIZE:
            if (wparam != 0) return 0;
            break;
        case WM_NCHITTEST:
            {
                if (platform->state.fullscreen) return HTCLIENT;
                POINT point{GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam)};
                ScreenToClient(window, &point);
                RECT client{};
                GetClientRect(window, &client);
                if (!IsZoomed(window)) {
                    const UINT dpi    = GetDpiForWindow(window);
                    const int border  = GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi) + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
                    const bool left   = point.x < border;
                    const bool right  = point.x >= client.right - border;
                    const bool top    = point.y < border;
                    const bool bottom = point.y >= client.bottom - border;
                    if (top && left) return HTTOPLEFT;
                    if (top && right) return HTTOPRIGHT;
                    if (bottom && left) return HTBOTTOMLEFT;
                    if (bottom && right) return HTBOTTOMRIGHT;
                    if (left) return HTLEFT;
                    if (right) return HTRIGHT;
                    if (top) return HTTOP;
                    if (bottom) return HTBOTTOM;
                }
                const auto& region = platform->drag_region;
                if (point.x >= region[0] && point.y >= region[1] && point.x < region[2] && point.y < region[3]) return HTCAPTION;
                return HTCLIENT;
            }
        case WM_GETMINMAXINFO:
            {
                MONITORINFO monitor_info{sizeof(MONITORINFO)};
                GetMonitorInfoW(MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST), &monitor_info);
                MINMAXINFO& minmax    = *reinterpret_cast<MINMAXINFO*>(lparam);
                const auto& area = platform->state.fullscreen ? monitor_info.rcMonitor : monitor_info.rcWork;
                minmax.ptMaxPosition  = {area.left - monitor_info.rcMonitor.left, area.top - monitor_info.rcMonitor.top};
                minmax.ptMaxSize      = {area.right - area.left, area.bottom - area.top};
                const auto dpi        = GetDpiForWindow(window);
                minmax.ptMinTrackSize = {MulDiv(960, dpi, 96), MulDiv(600, dpi, 96)};
                return 0;
            }
        case WM_SYSCOMMAND:
            if (platform->state.fullscreen && ((wparam & 0xFFF0) == SC_MOVE || (wparam & 0xFFF0) == SC_SIZE || (wparam & 0xFFF0) == SC_MAXIMIZE)) return 0;
            break;
        case WM_DPICHANGED:
            if (platform->state.fullscreen) {
                MONITORINFO monitor_info{sizeof(MONITORINFO)};
                GetMonitorInfoW(MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST), &monitor_info);
                return CallWindowProcW(platform->state.original_window_proc, window, message, wparam, reinterpret_cast<LPARAM>(&monitor_info.rcMonitor));
            }
            break;
        }
        return CallWindowProcW(platform->state.original_window_proc, window, message, wparam, lparam);
    }

} // namespace genesia::editor
