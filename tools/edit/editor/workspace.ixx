export module edit.editor.workspace;
export import edit.session;
import tools.editor.platform.window;
import std;
export namespace edit::editor {
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
        void save_prompt();

    private:
        std::string prompt;
        std::filesystem::path settings;
        std::optional<std::chrono::steady_clock::time_point> save_at;
        std::optional<bool> connected;
        std::chrono::steady_clock::time_point next_connection_check{};
        std::future<bool> connection_check;
    };
} // namespace edit::editor
