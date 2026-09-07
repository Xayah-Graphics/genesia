module;

#include <imgui.h>

export module genesia.editor.ui;

import genesia.generation.configuration;
import genesia.generation.output;
import genesia.sdxl;
import genesia.editor.platform.window;
import genesia.editor.platform.interop;
import genesia.editor.ui.renderer;
import genesia.editor.session;
import std;

export namespace genesia::editor {
    struct UserInterface final {
        struct History final {
            std::uint64_t id;
            Record record;
            std::uint64_t texture{};
            bool saved{};
        };

        Configuration configuration;
        std::filesystem::path configuration_path;
        WindowPlatform& window;
        Renderer& renderer;
        Interop& interop;
        Session& session;
        sdxl::Parameters draft;
        std::vector<History> history;
        std::uint64_t seed{};
        bool random_seed{true};
        bool composer_open{true};
        bool history_open{};
        bool advanced{};
        bool following_latest{true};
        bool edited_since_submit{};
        float composer_amount{1};
        float history_amount{};
        float zoom{1};
        bool fit_image{true};
        ImVec2 pan{};
        bool image_live{};
        float progress_alpha{};
        std::string progress_label;
        std::uint64_t observed_task{std::numeric_limits<std::uint64_t>::max()};
        std::uint32_t preview_step{};
        std::uint64_t transition_texture{};
        int transition_width{}, transition_height{};
        double transition_started{};
        std::uint64_t image_texture{};
        std::uint64_t selected{std::numeric_limits<std::uint64_t>::max()};
        int image_width{}, image_height{};
        double animate_until{};
        double refresh_at{std::numeric_limits<double>::infinity()};
        std::vector<double> display_times;
        std::string shown_error;

        UserInterface(Configuration settings, std::filesystem::path path, WindowPlatform& platform, Renderer& display, Interop& bridge, Session& generation);
        void receive();
        void show_image(std::uint64_t texture, int width, int height, bool transition);
        void submit();
        void canvas(float scale, ImVec2 size, float composer_height);
        void top_strip(float scale, ImVec2 size);
        void composer(float scale, ImVec2 size, float height);
        void history_drawer(float scale, ImVec2 size);
        void draw();
    };
} // namespace genesia::editor
