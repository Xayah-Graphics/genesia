module;
#include <imgui.h>
export module genesia.editor.ui.tag_editor;
import genesia.prompt;
import std;

export namespace genesia::editor {
    struct TagSuggestion final {
        std::uint32_t tag;
        std::string_view match;
        int rank;
    };
    struct TagSearch final {
        struct Posting {
            std::uint32_t gram, begin, count;
        };
        const prompt::Catalog& catalog;
        std::vector<prompt::CatalogKey> keys;
        std::vector<Posting> index;
        std::vector<std::uint32_t> postings;

        explicit TagSearch(const prompt::Catalog& catalog);
        std::vector<TagSuggestion> search(std::string_view query) const;
    };
    struct TagMove final {
        std::size_t from_group, from_tag, to_group, to_tag;
    };
    struct TagEditor;
    struct TagLayout final {
        struct Item final {
            std::string_view text;
            std::string weight;
            ImVec2 position;
            float width, weight_width;
            bool input;
        };
        std::vector<Item> items;
        float width, height;
        bool input_expanded{};

        TagLayout(const prompt::Group& group, const prompt::Catalog& catalog, float width, float scale, const TagEditor* editor = nullptr);
    };
    struct TagEditor final {
        std::uint32_t id{};
        bool background_hovered{};
        std::string input;
        std::string analyzed_input;
        std::size_t cursor{}, analyzed_cursor{}, completion_begin{}, completion_end{};
        std::optional<prompt::Error> error;
        std::set<std::size_t> selection;
        std::size_t anchor{};
        std::optional<std::size_t> editing;
        bool valid{true};
        bool menu_open{};
        bool focus_input{};
        bool input_active{};
        bool completion_requested{};
        bool select_error{};
        std::uint32_t revision{};
        int highlighted{};
        std::string query;
        std::vector<TagSuggestion> suggestions;

        void replace(prompt::Group& group, std::vector<prompt::Tag> tags);
        bool commit(prompt::Group& group, const prompt::Catalog& catalog, std::optional<std::uint32_t> candidate = {});
        void draw(const char* payload_type, std::size_t group_index, prompt::Group& group, const TagSearch& search, const TagLayout& layout, float scale, std::optional<TagMove>& move);
        static int input_callback(ImGuiInputTextCallbackData* data);
    };
    struct PromptEditor final {
        std::array<std::vector<TagEditor>, 2> groups;
        std::array<TagEditor, 2> additions;
        std::array<bool, 2> adding{};
        std::vector<prompt::Pair> undo, redo;
        std::uint32_t next_id{};
        bool negative_open{};
        bool valid{true};
        bool escape_owned{}, focus_input{};

        void reset(const prompt::Pair& prompt, bool clear_history = true);
        void suspend();
        void remember(prompt::Pair before, const prompt::Pair& after);
        bool commit(prompt::Pair& prompt, const prompt::Catalog& catalog);
        void draw_groups(prompt::Side& side, std::size_t side_index, const TagSearch& search, float scale);
        void draw(prompt::Pair& prompt, const TagSearch& search, float scale);
    };
    void view_prompt(const prompt::Pair& prompt, const prompt::Catalog& catalog, float scale);
} // namespace genesia::editor
