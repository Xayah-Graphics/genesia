export module foreground.editor.workspace;
export import foreground.session;
import tools.editor.platform.window;
import std;
export namespace foreground::editor {
    struct Workspace final {
        tools::editor::WindowPlatform& window;
        const float& dpi;
        runtime::Session session;
        runtime::Snapshot state;
        std::string error;
        explicit Workspace(tools::editor::WindowPlatform& window, const float& dpi);
        void receive();
        void drop();
        void draw();
    };
} // namespace foreground::editor
