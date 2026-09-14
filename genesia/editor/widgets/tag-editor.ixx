module;
#include <imgui.h>
export module genesia.editor.widgets.tags;
import genesia.prompt;
export import genesia.prompt.search;
import genesia.prompt.document;
import std;

export namespace genesia::editor {
    struct TagMove final {
        std::size_t from_group, from_tag, to_group, to_tag;
    };
    struct TagEditor;
    struct TagChange final {
        std::optional<prompt::Tag> original;
        std::size_t group{}, index{};
        bool removed{};
        bool operator==(const TagChange&) const = default;
    };
    struct GroupChange final {
        std::optional<std::size_t> original;
        bool enabled{true};
        std::vector<TagChange> tags;
        bool operator==(const GroupChange&) const = default;
    };
    struct TagLayout final {
        struct Item final {
            std::string_view text;
            std::string weight;
            ImVec2 position;
            float width, weight_width;
            bool input;
            bool added{};
        };
        std::vector<Item> items;
        float width, height;
        bool input_expanded{};

        TagLayout(const prompt::Group& group, const prompt::Catalog& catalog, float width, float scale, const TagEditor* editor);
    };
    struct TagEditor final {
        std::optional<GroupChange> change;
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
        std::vector<prompt::TagSuggestion> suggestions;

        void replace(prompt::Group& group, std::vector<prompt::Tag> tags);
        void erase(prompt::Group& group, std::size_t index);
        bool commit(prompt::Group& group, const prompt::Catalog& catalog, std::optional<std::uint32_t> candidate = {});
        void draw(const char* payload_type, std::size_t group_index, prompt::Group& group, const prompt::TagSearch& search, const prompt::Catalog& catalog, const TagLayout& layout, float scale, std::optional<TagMove>& move);
        static int input_callback(ImGuiInputTextCallbackData* data);
    };
    struct PromptEditor final {
        struct Snapshot final {
            prompt::Pair prompt;
            std::array<std::vector<std::optional<GroupChange>>, 2> changes;
            bool operator==(const Snapshot&) const = default;
        };
        std::array<std::vector<TagEditor>, 2> groups;
        std::array<TagEditor, 2> additions;
        std::array<bool, 2> adding{};
        std::vector<Snapshot> undo, redo;
        bool tracking{};
        std::uint32_t next_id{};
        bool negative_open{};
        bool valid{true};
        bool escape_owned{}, focus_input{};

        void reset(const prompt::Pair& prompt, bool clear_history = true);
        void suspend();
        Snapshot snapshot(const prompt::Pair& prompt) const;
        prompt::Pair materialize(const prompt::Pair& prompt) const;
        void remember(Snapshot before, const prompt::Pair& after);
        bool commit(prompt::Pair& prompt, const prompt::Catalog& catalog);
        void draw_groups(prompt::Side& side, std::size_t side_index, const prompt::TagSearch& search, const prompt::Catalog& catalog, float scale);
        void draw(prompt::Pair& prompt, const prompt::TagSearch& search, const prompt::Catalog& catalog, float scale);
    };
    void show_prompt(const prompt::Pair& prompt, const prompt::Catalog& catalog, float scale);
} // namespace genesia::editor
