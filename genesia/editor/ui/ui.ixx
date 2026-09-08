module;

#include <imgui.h>

export module genesia.editor.ui;

import genesia.generation.defaults;
import genesia.prompt.preset;
import genesia.generation.output;
import genesia.sdxl;
import genesia.editor.platform.window;
import genesia.editor.platform.interop;
import genesia.editor.ui.renderer;
import genesia.editor.ui.tag_editor;
import genesia.editor.session;
import std;

export namespace genesia::editor {
    struct UserInterface final {
        struct ImageView final {
            float zoom{1};
            bool fit{true};
            bool dragging{};
            ImVec2 center{0.5F, 0.5F};
            float initial_zoom{1}, target_zoom{1};
            ImVec2 anchor{}, pivot{};
            double started{-1};

            void scale_to(float ratio, ImVec2 position, ImVec2 image, bool fitting, double now);
            void update(ImVec2 available, ImVec2 image, double now);
            void constrain(ImVec2 available, ImVec2 image);
        };
        struct History final {
            std::uint64_t id;
            Record record;
            std::uint64_t texture{};
            bool saved{};
        };
        struct ParameterEdit final {
            ImGuiID id{};
            ImGuiDataType type{};
            void* value{};
        };
        struct ControlLayout final {
            float right_width, image_label_width;
            bool different, image_above;
        };

        const std::shared_ptr<const prompt::Catalog> catalog;
        prompt::Preset preset;
        std::vector<std::string> preset_names;
        std::string pending_preset;
        std::array<char, 128> new_preset_name{};
        bool save_as_requested{};
        std::string preset_error;
        bool preview_enabled{defaults::preview_enabled};
        WindowPlatform& window;
        Renderer& renderer;
        Interop& interop;
        Session& session;
        sdxl::Parameters draft;
        prompt::Pair prompt;
        TagSearch tag_search;
        PromptEditor prompt_editor;
        std::vector<History> history;
        std::uint64_t seed{defaults::seeds.front()};
        bool random_seed{defaults::random_seed};
        bool tags_open{};
        bool history_open{};
        ParameterEdit parameter_edit;
        bool following_latest{true};
        float tags_amount{};
        float history_amount{};
        ImageView view;
        bool image_live{};
        float progress_alpha{};
        std::string progress_label;
        std::string progress_time;
        std::uint64_t observed_task{std::numeric_limits<std::uint64_t>::max()};
        std::uint32_t preview_step{};
        std::uint64_t transition_texture{};
        int transition_width{}, transition_height{};
        double transition_started{};
        std::uint64_t image_texture{};
        std::uint64_t selected{std::numeric_limits<std::uint64_t>::max()};
        std::optional<std::uint64_t> requested_image;
        int image_width{}, image_height{};
        double animate_until{};
        double refresh_at{std::numeric_limits<double>::infinity()};
        std::string shown_error;
        bool escape_owned{};

        UserInterface(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, WindowPlatform& platform, Renderer& display, Interop& bridge, Session& generation);
        void receive();
        void show_image(std::uint64_t texture, int width, int height, std::uint64_t id, bool transition);
        bool prepare_prompt();
        void commit_parameters();
        bool save_prompt();
        void switch_preset();
        void preset_dialogs(float scale);
        void submit();
        void return_to_create();
        ControlLayout control_layout(float scale, ImVec2 size) const;
        void canvas(float scale, ImVec2 size);
        void generation_settings(float scale, ImVec2 size);
        void top_strip(float scale, ImVec2 size);
        void tag_column(float scale, ImVec2 size, const ControlLayout& layout);
        void bottom_controls(float scale, ImVec2 size, const ControlLayout& layout);
        void history_strip(float scale, ImVec2 size);
        void draw();
    };
} // namespace genesia::editor
