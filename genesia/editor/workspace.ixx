module;
#include <imgui.h>
export module genesia.editor.workspace;
import genesia.generation.settings;
import genesia.editor.panels.prompts;
import genesia.editor.platform.window;
import genesia.editor.graphics.renderer;
import genesia.editor.widgets.tags;
import genesia.editor.graphics.bridge;
import genesia.editor.viewing.textures;
import genesia.editor.viewing.camera;
import genesia.editor.widgets.controls;
import std;
export namespace genesia::editor {
    struct Workspace final {
        enum class Page { generation, history };
        enum class View { browse, inspect };
        struct Output final {
            std::optional<std::uint64_t> task;
            std::optional<Record> record;
            std::optional<std::filesystem::path> saved;
            std::uint64_t texture{};
            int width{}, height{};
            bool preview{};
            std::uint32_t step{};
        };
        struct Position final {
            std::filesystem::path selected;
            std::size_t index{};
            float scroll{};
        };
        struct Picture final {
            std::uint64_t texture{};
            int width{}, height{};
            const Record* record{};
            std::optional<std::filesystem::path> file;
            bool preview{};
            std::string_view record_error;
        };
        struct Sidebar final {
            bool open{};
            float amount{}, width{};
        };
        struct ControlLayout final {
            float right_width{}, image_label_width{};
            bool different{}, image_above{};
        };
        struct LoraSettings final {
            std::string file;
            bool active{};
            float weight{defaults::lora_weight}, start{defaults::lora_start * 100};
        };
        const std::shared_ptr<const prompt::Catalog> catalog;
        const std::shared_ptr<const prompts::Library> prompt_library;
        std::string preset_name;
        std::vector<std::string> preset_names;
        std::array<char, 128> new_preset_name{};
        bool save_as_requested{}, final_prompt_requested{};
        std::string preset_error;
        bool preview_enabled{defaults::preview_enabled};
        WindowPlatform& window;
        Renderer& renderer;
        PromptPanel prompt_panel;
        WorkspaceRuntime runtime;
        TextureCache textures;
        generation::Settings draft;
        prompts::Recipe prompt;
        prompt::TagSearch tag_search;
        PromptEditor prompt_editor;
        Output generation;
        std::vector<LoraSettings> loras;
        runtime::Snapshot session_state;
        std::optional<runtime::TaskStatus> task;
        Page page{Page::generation};
        View viewing{View::browse};
        Position position;
        std::optional<std::filesystem::path> pending_delete;
        std::uint64_t seed{defaults::seed};
        bool random_seed{defaults::random_seed}, continuous_generation{};
        Sidebar prompt_sidebar;
        ImVec2 canvas_size{};
        ParameterEdit parameter_edit;
        ImageView view, generation_view;
        ImVec2 view_available{};
        float progress_alpha{};
        std::string progress_label, progress_time;
        double frame_time{}, animate_until{};
        double refresh_at{std::numeric_limits<double>::infinity()};
        std::string shown_error, action_error;

        Workspace(prompts::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, std::shared_ptr<const prompts::Library> prompt_library, WindowPlatform& platform, Renderer& renderer);
        ~Workspace();
        void receive();
        void update_generation();
        void update_web();
        void open_history();
        void center_image(std::size_t index);
        void back();
        Picture resolve_image(const std::filesystem::path& file) const;
        Picture resolve_generation() const;
        void commit_parameters();
        bool prepare_prompt();
        bool save_prompt();
        void switch_preset(std::string name);
        void begin_generation(std::uint64_t task, int width, int height);
        generation::Settings parameters() const;
        bool submit();
        void draw();
    };
} // namespace genesia::editor
