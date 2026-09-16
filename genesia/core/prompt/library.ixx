export module genesia.prompt.library;
export import genesia.prompt;
import std;

export namespace genesia::prompts {
    struct Choices final {
        std::string name, initial;
        std::vector<std::string> order;
        std::map<std::string, std::array<std::string, 2>> options;
    };
    struct Character final {
        std::string description;
        std::vector<std::string> hidden;
        std::vector<Choices> parts;
    };
    struct Rule final {
        std::string reason;
        std::map<std::string, std::vector<std::string>> when;
        std::vector<std::string> disable, require;
    };
    struct Scene final {
        std::string category, description;
        std::array<std::string, 2> text;
        std::vector<Choices> variations;
        std::vector<Rule> rules;
    };
    struct Library final {
        std::filesystem::path directory;
        std::map<std::string, Character> characters;
        std::map<std::string, Scene> scenes;

        Library(std::filesystem::path directory, const prompt::Catalog& catalog);
    };
    struct Recipe final {
        std::string character;
        std::map<std::string, std::string> parts;
        std::optional<std::string> scene;
        std::map<std::string, std::string> variations;
        prompt::Pair free;
        bool operator==(const Recipe&) const = default;
    };
    struct Preset final {
        std::string name;
        Recipe recipe;
    };
    struct Composition final {
        struct Part final {
            bool hidden{};
            std::vector<std::size_t> disable, require;
        };
        std::array<std::string, 2> text;
        std::map<std::string, Part> parts;
        std::string error;
    };

    void select_character(const Library& library, Recipe& recipe, std::string character);
    void select_scene(const Library& library, Recipe& recipe, std::optional<std::string> scene);
    Composition compose(const Library& library, const Recipe& recipe, const prompt::Catalog& catalog);
    std::array<std::string, 2> card_prompt(const Library& library, const Recipe& recipe, const Composition& composition, bool scene);
    Preset read_preset(const Library& library, std::string name, const prompt::Catalog& catalog);
    void write_preset(const Library& library, const Preset& preset, const prompt::Catalog& catalog, bool replace = true);
    std::vector<std::string> list_presets(const std::filesystem::path& directory);
    void export_prompt(const std::filesystem::path& directory, std::string_view name, const Composition& composition);
} // namespace genesia::prompts
