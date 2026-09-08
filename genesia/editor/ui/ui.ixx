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
import genesia.editor.gallery;
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
        struct RepaintDraft final {
            Record original;
            std::uint64_t modified;
            prompt::Pair prompt;
            PromptEditor editor;

            RepaintDraft(const Record& original, std::uint64_t modified);
        };
        struct Thumbnail final {
            Gallery::File file;
            std::uint64_t texture{};
            int width{}, height{};
            std::uint64_t touched{};
            std::string error;
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
        Gallery gallery;
        sdxl::Parameters draft;
        prompt::Pair prompt;
        TagSearch tag_search;
        PromptEditor prompt_editor;
        std::map<std::uint64_t, Thumbnail> thumbnails;
        std::map<std::uint64_t, std::unique_ptr<RepaintDraft>> repaints;
        std::optional<Record> image_record;
        std::optional<Gallery::File> image_file;
        std::uint64_t thumbnail_clock{};
        std::size_t thumbnail_bytes{};
        bool initial_gallery{true};
        bool reveal_selected{};
        float gallery_scroll{};
        std::uint64_t seed{defaults::seeds.front()};
        bool random_seed{defaults::random_seed};
        bool tags_open{};
        bool gallery_open{};
        ParameterEdit parameter_edit;
        bool following_latest{true};
        bool image_tags{};
        float denoise{defaults::denoise};
        float tags_amount{};
        float gallery_amount{};
        ImageView view;
        bool image_live{};
        float progress_alpha{};
        std::string progress_label;
        std::string progress_time;
        std::uint64_t observed_task{std::numeric_limits<std::uint64_t>::max()};
        std::optional<std::uint64_t> displayed_task, completed_task, saved_task;
        std::filesystem::path latest_path;
        std::uint32_t preview_step{};
        std::uint64_t transition_texture{};
        int transition_width{}, transition_height{};
        double transition_started{};
        std::uint64_t image_texture{};
        std::uint64_t selected{};
        std::optional<Gallery::File> requested_image;
        std::uint64_t requested_ticket{};
        int image_width{}, image_height{};
        double animate_until{};
        double refresh_at{std::numeric_limits<double>::infinity()};
        std::string shown_error;
        bool escape_owned{};

        UserInterface(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, WindowPlatform& platform, Renderer& display, Interop& bridge, Session& generation);
        void receive();
        void show_image(std::uint64_t texture, int width, int height, bool transition, bool reset);
        bool select_image(const Gallery::File& file);
        void navigate_image(int direction);
        bool prepare_prompt();
        void commit_parameters();
        bool save_prompt();
        void switch_preset();
        void preset_dialogs(float scale);
        RepaintDraft& image_prompt();
        void submit(bool repaint = false);
        ControlLayout control_layout(float scale, ImVec2 size) const;
        void canvas(float scale, ImVec2 size);
        void generation_settings(float scale, ImVec2 size);
        void top_strip(float scale, ImVec2 size);
        void tag_column(float scale, ImVec2 size, const ControlLayout& layout);
        void bottom_controls(float scale, ImVec2 size, const ControlLayout& layout);
        void gallery_strip(float scale, ImVec2 size);
        void draw();
    };
} // namespace genesia::editor
