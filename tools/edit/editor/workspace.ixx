module;
#include <imgui.h>
export module edit.editor.workspace;
export import edit.session;
import edit.preset;
import tools.editor.platform.window;
import tools.editor.graphics.renderer;
import tools.images;
import std;
export namespace edit::editor {
    struct Workspace final {
        tools::editor::WindowPlatform& window;
        tools::editor::Renderer& renderer;
        runtime::Session session;
        runtime::Snapshot state;
        std::string error;
        std::string preset_name;
        explicit Workspace(tools::editor::WindowPlatform& window, tools::editor::Renderer& renderer);
        void receive();
        void drop();
        void draw();
        void save_preset(const std::string& name, bool replace = true);

    private:
        struct ImageSlot final {
            std::filesystem::path path;
            tools::Image thumbnail;
            std::uint64_t texture{};
            std::string error;
        };
        std::vector<ImageSlot> references;
        std::vector<std::uint64_t> retired;
        std::vector<std::array<float, 4>> drop_bounds;
        std::array<float, 4> reference_bounds{};
        bool reveal_reference{};
        std::string prompt;
        std::vector<std::string> preset_names;
        std::array<char, 128> new_preset_name{};
        std::string preset_error;
        std::optional<std::chrono::steady_clock::time_point> save_at;
        void load_preset(Preset preset);
        void load_image(ImageSlot& image);
        // 0: main input; 1..9: fixed slots; empty: fixed-area gap.
        std::optional<std::size_t> drop_target(float x, float y) const;
        bool draw_image(ImageSlot* image, std::size_t number, ImVec2 size);
    };
} // namespace edit::editor
