export module genesia.prompt.library;
export import genesia.prompt;
import std;

export namespace genesia::prompts {
    struct Part final {
        std::string name, initial;
        std::vector<std::string> order;
        std::map<std::string, std::array<std::string, 2>> options;
    };
    struct Category final {
        std::vector<std::string> order;
        std::map<std::string, std::array<std::string, 2>> options;
    };
    struct Character final {
        std::string description;
        std::vector<Part> parts;
    };
    struct Rule final {
        std::string reason;
        std::map<std::string, std::vector<std::optional<std::string>>> when;
        std::vector<std::string> disable;
    };
    struct Library final {
        std::filesystem::path directory;
        std::vector<std::string> character_order, category_order;
        std::map<std::string, Character> characters;
        std::map<std::string, Category> categories;
        std::vector<Rule> rules;

        explicit Library(std::filesystem::path directory);
    };
    struct Recipe final {
        std::string character;
        std::map<std::string, std::string> parts;
        std::map<std::string, std::optional<std::string>> categories;
        prompt::Pair free;
        bool operator==(const Recipe&) const = default;
    };
    struct Preset final {
        std::string name;
        Recipe recipe;
    };
    struct Composition final {
        std::array<std::string, 2> text;
        std::map<std::string, std::vector<std::size_t>> disabled;
    };

    void select_character(const Library& library, Recipe& recipe, std::string character);
    Composition compose(const Library& library, const Recipe& recipe, const prompt::Catalog& catalog);
    Preset read_preset(const std::filesystem::path& directory, std::string name, const prompt::Catalog& catalog);
    void write_preset(const std::filesystem::path& directory, const Preset& preset, const prompt::Catalog& catalog, bool replace = true);
    std::vector<std::string> list_presets(const std::filesystem::path& directory);
    void export_prompt(const std::filesystem::path& directory, std::string_view name, const Composition& composition);
} // namespace genesia::prompts
