module;
#include <imgui.h>
export module genesia.editor.workspace;
import genesia.generation.defaults;
import genesia.prompt.preset;
import genesia.generation.output;
import genesia.editor.platform.window;
import genesia.editor.graphics.interop;
import genesia.editor.graphics.renderer;
import genesia.editor.widgets.tags;
import genesia.editor.widgets.captions;
import genesia.editor.graphics.bridge;
import genesia.editor.viewing.textures;
import genesia.runtime.catalog;
import genesia.editor.viewing.camera;
import genesia.editor.widgets.controls;
import std;

export namespace genesia::editor {
    struct Workspace final {
        struct RepaintDraft final {
            prompt::Resolved document;
            PromptEditor editor;
            RepaintDraft(const std::optional<Record>& source, std::shared_ptr<const prompt::Catalog> catalog);
        };
        struct TrainingDraft final {
            int steps{400};
            training::Config config;
            std::optional<training::Metrics> metrics;
            std::vector<float> losses;
            bool restart_confirm{};
            explicit TrainingDraft(const training::TrainingData& source);
        };
        enum class Page { generation, dataset, audit };
        enum class ConceptTool { none, train, audit, classify, tags, export_dataset, model };
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
            std::optional<float> confidence;
        };
        struct ImageResult final {
            std::string summary, annotation;
            std::vector<std::string> distribution;
            bool failed{};
        };
        struct Sidebar final {
            bool open{};
            float amount{}, width{};
        };
        struct ControlLayout final {
            float right_width{}, image_label_width{};
            bool different{}, image_above{};
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
        runtime::CatalogState library;
        TextureCache textures;
        WorkspaceRuntime runtime;
        generation::Settings draft;
        prompt::Pair prompt;
        prompt::TagSearch tag_search;
        PromptEditor prompt_editor;
        std::map<std::string, std::unique_ptr<RepaintDraft>> repaints;
        Output generation;
        std::optional<Repaint> repaint;
        std::vector<std::string> activated;
        struct LoraControls final {
            std::array<char, 2048> path{};
            float weight{1};
            bool active{};
        };
        std::map<std::string, LoraControls> lora_controls;
        std::map<std::string, std::string> prediction_errors;
        std::vector<dataset::File> visible_images;
        std::string audit_key, audit_category;
        ConceptTool concept_tool{ConceptTool::none};
        bool choosing_type{};
        std::string type_error;
        std::map<std::string, TrainingDraft> training_drafts;
        std::map<std::string, std::array<char, 2048>> classify_paths;
        std::map<std::string, std::string> export_paths;
        CaptionEditor caption_editor;
        std::function<void()> caption_continuation;
        std::string caption_folder{"."};
        dataset::Collection folder_collection;
        classification::Audit audit_report;
        classification::Cache prediction_cache;
        runtime::Snapshot session_state;
        std::map<std::pair<std::string, runtime::Kind>, runtime::TaskStatus> activity;
        std::optional<std::uint64_t> audit_task;
        dataset::Collection audit_collection;
        Position audit_position;
        bool audit_dirty{};
        Page audit_return{Page::dataset};
        std::string audit_return_collection;
        View audit_return_view{View::browse};
        ImageView audit_return_camera;
        Page page{Page::generation};
        View viewing{View::browse};
        std::string collection_key;
        const dataset::Root* root{};
        const dataset::Collection* collection{};
        std::map<std::string, Position> positions;
        std::string locate_sha;
        std::optional<dataset::File> pending_delete;
        std::uint64_t seed{defaults::seed};
        bool random_seed{defaults::random_seed};
        Sidebar dataset_sidebar, prompt_sidebar;
        bool expand_dataset_roots{true};
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

        Workspace(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, WindowPlatform& platform, Renderer& renderer, std::string dataset);
        ~Workspace();
        void receive();
        void synchronize_collection();
        void select_collection(std::string key, std::string sha = {});
        void select_tool(ConceptTool tool);
        bool save_caption(std::function<void()> continuation = {});
        Position& current_position();
        void center_image(std::size_t index);
        void start_repaint(const dataset::File& source);
        bool leave_repaint();
        void back();
        Picture resolve_image(const dataset::File& file, Role role = Role::image) const;
        Picture resolve_output(const Output& output, Role role = Role::image) const;
        void commit_parameters();
        bool save_prompt();
        void switch_preset();
        void begin_output(Output& output, std::uint64_t task, int width, int height);
        void submit();

        void open_audit(std::string key, bool refresh = false);
        std::uint64_t submit_task(runtime::Request request);
        void task_event(const runtime::TaskStatus& event);
        void rebuild_audit();
        void observe_inference();
        void draw();
    };
} // namespace genesia::editor
