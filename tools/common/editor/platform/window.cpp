module;

#include <Windows.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <dwmapi.h>
#include <ole2.h>
#include <shellapi.h>

module tools.editor.platform.window;

import std;
import vulkan;

namespace tools::editor {
    WindowPlatform::WindowPlatform(const std::string_view application_name, const vk::Extent2D initial_extent) {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        this->state.glfw_window.reset(glfwCreateWindow(static_cast<int>(initial_extent.width), static_cast<int>(initial_extent.height), std::string{application_name}.c_str(), nullptr, nullptr));
        if (!this->state.glfw_window) throw std::runtime_error(std::format("{} window creation failed", application_name));
        this->window        = this->state.glfw_window.get();
        this->native_window = glfwGetWin32Window(this->window);
        glfwSetWindowUserPointer(this->window, this);
        glfwSetWindowCloseCallback(this->window, [](GLFWwindow* window) {
            WindowPlatform& platform       = *static_cast<WindowPlatform*>(glfwGetWindowUserPointer(window));
            platform.state.close_requested = true;
            glfwSetWindowShouldClose(window, GLFW_FALSE);
        });

        SetPropW(this->native_window, L"GenesiaToolWindow", this);
        this->state.original_window_proc = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(this->native_window, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&WindowPlatform::window_proc)));
        constexpr LONG_PTR style         = WS_POPUP | WS_MINIMIZEBOX | WS_SYSMENU;
        SetWindowLongPtrW(this->native_window, GWL_STYLE, style);
        constexpr BOOL dark_mode = TRUE;
        DwmSetWindowAttribute(this->native_window, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark_mode, sizeof(dark_mode));
        constexpr DWM_WINDOW_CORNER_PREFERENCE corners = DWMWCP_ROUND;
        DwmSetWindowAttribute(this->native_window, DWMWA_WINDOW_CORNER_PREFERENCE, &corners, sizeof(corners));
        RECT bounds{};
        GetWindowRect(this->native_window, &bounds);
        SetWindowPos(this->native_window, nullptr, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
        resize({static_cast<std::uint32_t>(bounds.right - bounds.left), static_cast<std::uint32_t>(bounds.bottom - bounds.top)});
        DragAcceptFiles(native_window, FALSE);
        const auto registered = RegisterDragDrop(native_window, &drop_target);
        if (FAILED(registered)) throw std::runtime_error{std::format("Register file drop: 0x{:08X}", static_cast<unsigned long>(registered))};
    }

    WindowPlatform::~WindowPlatform() {
        RevokeDragDrop(native_window);
        SetWindowLongPtrW(this->native_window, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(this->state.original_window_proc));
        RemovePropW(this->native_window, L"GenesiaToolWindow");
    }

    bool WindowPlatform::take_close_request() noexcept {
        return std::exchange(this->state.close_requested, false);
    }

    void WindowPlatform::move() {
        if (!(GetAsyncKeyState(VK_LBUTTON) & 0x8000)) return;
        POINT cursor{};
        GetCursorPos(&cursor);
        ReleaseCapture();
        SendMessageW(native_window, WM_NCLBUTTONDOWN, HTCAPTION, MAKELPARAM(cursor.x, cursor.y));
        // The native move loop consumes the release; also release GLFW's input state.
        GetCursorPos(&cursor);
        ScreenToClient(native_window, &cursor);
        SendMessageW(native_window, WM_LBUTTONUP, 0, MAKELPARAM(cursor.x, cursor.y));
        redraw = true;
    }

    void WindowPlatform::resize(const vk::Extent2D extent) {
        MONITORINFO monitor{sizeof(MONITORINFO)};
        GetMonitorInfoW(MonitorFromWindow(native_window, MONITOR_DEFAULTTONEAREST), &monitor);
        const auto& area  = monitor.rcWork;
        const auto margin = MulDiv(16, GetDpiForWindow(native_window), 96);
        const vk::Extent2D limit{static_cast<std::uint32_t>(area.right - area.left - 2 * margin), static_cast<std::uint32_t>(area.bottom - area.top - 2 * margin)};
        if (extent_limit != limit) {
            extent_limit = limit;
            redraw       = true;
        }
        if (IsIconic(native_window)) return;
        const auto width  = static_cast<int>(std::min(extent.width, limit.width));
        const auto height = static_cast<int>(std::min(extent.height, limit.height));
        RECT bounds{};
        GetWindowRect(native_window, &bounds);
        const bool visible = IsWindowVisible(native_window) != FALSE;
        const auto x       = visible ? std::clamp(static_cast<int>(bounds.left), static_cast<int>(area.left) + margin, static_cast<int>(area.right) - margin - width) : area.left + (area.right - area.left - width) / 2;
        const auto y       = visible ? std::clamp(static_cast<int>(bounds.top), static_cast<int>(area.top) + margin, static_cast<int>(area.bottom) - margin - height) : area.top + (area.bottom - area.top - height) / 2;
        if (bounds.left == x && bounds.top == y && bounds.right - bounds.left == width && bounds.bottom - bounds.top == height) return;
        SetWindowPos(native_window, nullptr, x, y, width, height, SWP_NOZORDER | SWP_NOACTIVATE);
        redraw = true;
    }

    WindowPlatform::GlfwLifetime::GlfwLifetime() {
        if (glfwInit() != GLFW_TRUE) throw std::runtime_error("GLFW initialization failed");
    }

    WindowPlatform::GlfwLifetime::~GlfwLifetime() {
        glfwTerminate();
    }

    WindowPlatform::OleLifetime::OleLifetime() {
        const auto result = OleInitialize(nullptr);
        if (FAILED(result)) throw std::runtime_error{std::format("Initialize file drag: 0x{:08X}", static_cast<unsigned long>(result))};
    }

    WindowPlatform::OleLifetime::~OleLifetime() {
        OleUninitialize();
    }

    WindowPlatform::DropTarget::DropTarget(WindowPlatform& owner) : window{owner} {}

    HRESULT STDMETHODCALLTYPE WindowPlatform::DropTarget::QueryInterface(REFIID iid, void** object) {
        if (iid == __uuidof(IUnknown) || iid == __uuidof(IDropTarget)) {
            *object = static_cast<IDropTarget*>(this);
            AddRef();
            return S_OK;
        }
        *object = nullptr;
        return E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE WindowPlatform::DropTarget::AddRef() {
        return ++references;
    }

    ULONG STDMETHODCALLTYPE WindowPlatform::DropTarget::Release() {
        // Owned by WindowPlatform; RevokeDragDrop releases OLE's reference before destruction.
        return --references;
    }

    HRESULT STDMETHODCALLTYPE WindowPlatform::DropTarget::DragEnter(IDataObject* data, DWORD keys, POINTL point, DWORD* effect) {
        window.dragged.clear();
        window.drop_error.clear();
        FORMATETC format{CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL};
        STGMEDIUM storage{};
        if (SUCCEEDED(data->GetData(&format, &storage))) {
            try {
                const auto handle = static_cast<HDROP>(storage.hGlobal);
                const auto count  = DragQueryFileW(handle, 0xFFFFFFFF, nullptr, 0);
                for (UINT i = 0; i < count; ++i) {
                    std::wstring path(DragQueryFileW(handle, i, nullptr, 0) + 1, L'\0');
                    const auto length = DragQueryFileW(handle, i, path.data(), static_cast<UINT>(path.size()));
                    path.resize(length);
                    window.dragged.emplace_back(std::move(path));
                }
            } catch (const std::exception& failure) {
                window.drop_error = failure.what();
                window.dragged.clear();
            }
            ReleaseStgMedium(&storage);
        }
        return DragOver(keys, point, effect);
    }

    HRESULT STDMETHODCALLTYPE WindowPlatform::DropTarget::DragOver(DWORD, POINTL point, DWORD* effect) {
        window.drag_position = {point.x, point.y};
        ScreenToClient(window.native_window, &window.drag_position);
        *effect       = window.dragged.empty() ? DROPEFFECT_NONE : *effect & DROPEFFECT_COPY;
        window.redraw = true;
        glfwPostEmptyEvent();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE WindowPlatform::DropTarget::DragLeave() {
        window.dragged.clear();
        window.redraw = true;
        glfwPostEmptyEvent();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE WindowPlatform::DropTarget::Drop(IDataObject*, DWORD keys, POINTL point, DWORD* effect) {
        DragOver(keys, point, effect);
        if (*effect & DROPEFFECT_COPY) {
            window.dropped       = std::move(window.dragged);
            window.drop_position = window.drag_position;
        }
        window.dragged.clear();
        return S_OK;
    }

    LRESULT CALLBACK WindowPlatform::window_proc(HWND window, const UINT message, const WPARAM wparam, const LPARAM lparam) {
        WindowPlatform* platform = static_cast<WindowPlatform*>(GetPropW(window, L"GenesiaToolWindow"));
        if (platform == nullptr) return DefWindowProcW(window, message, wparam, lparam);
        if ((message >= WM_MOUSEFIRST && message <= WM_MOUSELAST) || (message >= WM_KEYFIRST && message <= WM_KEYLAST) || message == WM_MOUSELEAVE || message == WM_MOVE || message == WM_SIZE || message == WM_DPICHANGED || message == WM_DISPLAYCHANGE || message == WM_SETTINGCHANGE || message == WM_PAINT || message == WM_SETFOCUS || message == WM_KILLFOCUS) platform->redraw = true;
        switch (message) {
        case WM_KEYDOWN:
            if (wparam == 'W' && GetKeyState(VK_CONTROL) < 0 && GetKeyState(VK_SHIFT) >= 0 && GetKeyState(VK_MENU) >= 0 && GetKeyState(VK_LWIN) >= 0 && GetKeyState(VK_RWIN) >= 0) {
                if (!(lparam & (1LL << 30))) SendMessageW(window, WM_CLOSE, 0, 0);
                return 0;
            }
            break;
        case WM_NCCALCSIZE:
            if (wparam != 0) return 0;
            break;
        case WM_NCHITTEST: return HTCLIENT;
        }
        return CallWindowProcW(platform->state.original_window_proc, window, message, wparam, lparam);
    }

} // namespace tools::editor
