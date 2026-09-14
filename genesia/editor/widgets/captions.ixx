export module genesia.editor.widgets.captions;
import genesia.prompt.search;
import std;
export namespace genesia::editor {
    struct CaptionEditor final {
        std::string key, input, saved, query, error;
        std::vector<prompt::TagSuggestion> suggestions;
        std::optional<std::uint64_t> task;
        double saved_at{};
        int cursor{};
        bool editing{}, focus{};
        bool draw(const prompt::TagSearch& search, float scale);
    };
} // namespace genesia::editor
