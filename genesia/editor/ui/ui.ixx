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
import genesia.editor.library;
import std;

export namespace genesia::editor {
    struct UserInterface final {
        struct ImageView final {
            float zoom{1};
            bool fit{true}, dragging{};
            ImVec2 center{0.5F, 0.5F};
            float initial_zoom{1}, target_zoom{1};
            ImVec2 anchor{}, pivot{};
            double started{-1};
            void scale_to(float ratio, ImVec2 position, ImVec2 image, bool fitting, double now);
            void update(ImVec2 available, ImVec2 image, double now);
            void constrain(ImVec2 available, ImVec2 image);
        };
        struct RepaintDraft final {
            const std::shared_ptr<const prompt::Catalog> catalog;
            prompt::Pair prompt;
            PromptEditor editor;
            explicit RepaintDraft(const Record& source);
        };
        enum class Page { generation, dataset };
        enum class View { browse, inspect, repaint, comparison, source, result };
        enum class Role { image, source, result };
        enum class ImageAction { none, click, repaint };
        struct Output final {
            std::optional<std::uint64_t> task;
            std::optional<Record> record;
            std::optional<dataset::File> saved;
            std::uint64_t texture{};
            int width{}, height{};
            bool preview{};
            std::uint32_t step{};
        };
        struct Repaint final {
            dataset::File source;
            Page return_page;
            View return_view;
            ImageView return_camera;
            Output result;
        };
        struct Position final {
            std::string selected;
            std::size_t index{};
            float scroll{};
        };
        struct Picture final {
            std::uint64_t texture{};
            int width{}, height{};
            const Record* record{};
            std::optional<dataset::File> file;
            bool preview{};
            Role role{Role::image};
        };
        struct ParameterEdit final {
            ImGuiID id{};
            ImGuiDataType type{};
            void* value{};
        };
        struct Sidebar final {
            bool open{};
            float amount{}, width{};
        };
        struct ControlLayout final {
            struct Classifier final {
                std::string_view id;
                const classifier::Result* result{};
                std::string verdict;
                ImVec4 ink;
                bool available{}, enabled{};
            };
            float right_width{}, image_label_width{};
            bool different{}, image_above{};
            std::vector<Classifier> classifiers;
            float classifier_width{}, classifier_height{}, classifier_bottom{};
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
        Library library;
        const std::vector<classifier::Descriptor> classifiers{classifier::discover()};
        classifier::Selection classification;
        std::unique_ptr<GenerationRuntime> runtime;
        sdxl::Parameters draft;
        prompt::Pair prompt;
        TagSearch tag_search;
        PromptEditor prompt_editor;
        std::map<std::string, std::unique_ptr<RepaintDraft>> repaints;
        Output generation;
        std::optional<Repaint> repaint;
        Page page{Page::generation};
        View viewing{View::browse};
        std::string collection_key;
        dataset::Root* root{};
        dataset::Collection* collection{};
        std::map<std::string, Position> positions;
        std::optional<std::filesystem::path> locate;
        std::uint64_t seed{defaults::seed};
        bool random_seed{defaults::random_seed};
        Sidebar dataset_sidebar, prompt_sidebar;
        ImVec2 canvas_origin{}, canvas_size{};
        ParameterEdit parameter_edit;
        float denoise{defaults::denoise};
        ImageView view, generation_view;
        ImVec2 view_available{};
        float progress_alpha{};
        std::string progress_label, progress_time;
        double frame_time{}, animate_until{};
        double refresh_at{std::numeric_limits<double>::infinity()};
        std::string shown_error, action_error;
        bool escape_owned{};

        UserInterface(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, WindowPlatform& platform, Renderer& renderer, std::string dataset);
        ~UserInterface();
        void receive();
        void synchronize_collection();
        void select_collection(std::string key, std::optional<std::filesystem::path> locate = {});
        void center_image(std::size_t index);
        void start_repaint(const dataset::File& source);
        bool leave_repaint();
        void back();
        Picture resolve_image(const dataset::File& file, Role role = Role::image) const;
        Picture resolve_output(const Output& output, Role role = Role::image) const;
        void commit_parameters();
        bool save_prompt();
        void switch_preset();
        void preset_dialogs(float scale);
        void begin_output(Output& output, std::uint64_t task, int width, int height);
        void submit();
        ControlLayout control_layout(float scale, ImVec2 size, const Picture& image) const;
        ImageAction image_panel(const char* id, const Picture& image, ImVec2 origin, ImVec2 size, float scale, bool interactive, float brightness = 1);
        void canvas(float scale, ImVec2 size);
        void generation_settings(float scale, ImVec2 size);
        void top_strip(float scale, ImVec2 size);
        void sidebar(bool left, float scale, ImVec2 size, const Picture& image);
        std::optional<std::string> dataset_contents();
        void bottom_controls(float scale, ImVec2 size, const ControlLayout& layout, const Picture& image);
        void classifier_controls(float scale, ImVec2 size, const ControlLayout& layout, const Picture& image);
        void draw();
    };
} // namespace genesia::editor
