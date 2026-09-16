export module genesia.editor.panels.prompts;
export import genesia.prompt.library;
import genesia.editor.prompt.previews;
import genesia.editor.platform.window;
import genesia.editor.graphics.renderer;
import std;

export namespace genesia::editor {
    struct PromptPanel final {
        const prompts::Library& library;
        previews::Images images;
        prompts::Composition composition;
        std::optional<std::filesystem::path> incoming;
        std::string error;
        bool ready{};

        PromptPanel(const prompts::Library& library, Renderer& renderer, WindowPlatform& window);
        void update(const prompts::Recipe& recipe, const prompt::Catalog& catalog);
        void status();
        void draw(prompts::Recipe& recipe, const prompt::Catalog& catalog, float scale);

    private:
        WindowPlatform& window;
        std::optional<prompts::Recipe> evaluated;
        std::optional<previews::Location> location;
        std::filesystem::path scene_preview, hovered_preview;
        int hover_frame{-1};
        std::array<std::array<char, 256>, 2> selection_search{};

        void selection_preview(const std::string& name, bool scene, const prompts::Recipe& recipe, float scale, const prompts::Choices* group = nullptr, const std::string* suboption = nullptr);
        void picture(const std::filesystem::path& path, const std::string& name, float width, float height, float scale, bool scene);
        void option_text(const std::array<std::string, 2>& text, const std::string& name, float scale, float width, const prompts::Composition::Part* effect = nullptr) const;
    };
} // namespace genesia::editor
