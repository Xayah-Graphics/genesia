export module qwen.editor.workspace;
export import qwen.session;
import tools.editor.platform.window;
import std;
export namespace qwen::editor {
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
    private:
        std::string prompt;
        std::vector<std::string> choices{"YES", "NO"};
        std::optional<std::size_t> editing_choice;
        std::string choice_original;
        bool focus_choice{};
        bool choice_created{};
        std::vector<Item> results;
        std::filesystem::path settings;
        std::optional<std::chrono::steady_clock::time_point> save_at;
        std::optional<bool> connected;
        std::chrono::steady_clock::time_point next_connection_check{};
        std::future<bool> connection_check;
        void save_settings();
    };
} // namespace qwen::editor
