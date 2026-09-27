export module genesia.prompt;
export import genesia.prompt.catalog;
import std;

export namespace genesia::prompt {
    struct Tag final {
        std::uint32_t id;
        float weight{1};
    };
    struct Group final {
        std::vector<Tag> tags;
        bool enabled{true};
    };
    struct Side final {
        std::vector<Group> groups;
        std::string fixed;
    };
    struct Pair final {
        Side positive, negative;
    };
    struct Input final {
        std::string name;
        float weight{1};
    };
    struct Error final {
        std::string message;
        std::size_t offset{}, length{};
    };
    std::expected<Input, Error> parse_tag(const Catalog& catalog, std::string_view text, bool completion = false);
    std::expected<std::vector<Tag>, Error> parse(const Catalog& catalog, std::string_view text);
    std::string serialize(const Catalog& catalog, std::span<const Tag> tags, bool conditioning = false);
    std::string compose(const Catalog& catalog, const Side& side);
} // namespace genesia::prompt
