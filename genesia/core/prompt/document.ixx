export module genesia.prompt.document;
export import genesia.prompt;
import std;
export namespace genesia::prompt {
    struct StoredTag final {
        std::string name, text;
        float weight{1};
    };
    struct StoredGroup final {
        bool enabled{true};
        std::vector<StoredTag> tags;
    };
    struct StoredSide final {
        std::string text, fixed;
        std::vector<StoredGroup> groups;
    };
    struct Document final {
        std::array<StoredSide, 2> sides;
    };
    struct Resolved final {
        std::shared_ptr<const Catalog> catalog;
        Pair prompt;
    };
    Document store(const Pair& prompt, const Catalog& catalog);
    Resolved resolve(const Document& document, std::shared_ptr<const Catalog> catalog);
} // namespace genesia::prompt
