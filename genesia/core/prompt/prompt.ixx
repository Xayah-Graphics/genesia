export module genesia.prompt;
export import genesia.prompt.catalog;
import std;

export namespace genesia::prompt {
    struct Tag final {
        std::uint32_t id;
        float weight{1};
        bool operator==(const Tag&) const = default;
    };
    struct Group final {
        std::vector<Tag> tags;
        bool enabled{true};
        bool operator==(const Group&) const = default;
    };
    struct Side final {
        std::vector<Group> groups;
        std::string fixed;
        bool operator==(const Side&) const = default;
    };
    struct Pair final {
        Side positive, negative;
        bool operator==(const Pair&) const = default;
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
}
