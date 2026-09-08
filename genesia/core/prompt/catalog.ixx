export module genesia.prompt.catalog;
import std;

export namespace genesia::prompt {
    struct CustomTag final {
        std::string name;
        std::string text;
    };
    struct CatalogTag final {
        std::string_view name;
        std::string_view text;
        std::uint32_t count;
        int category;
    };
    struct CatalogKey final {
        std::string_view name;
        std::uint32_t tag;
    };
    struct Catalog final {
        std::string storage;
        std::vector<CatalogTag> tags;
        std::vector<CatalogKey> names;
        std::vector<CatalogKey> alias_names;

        explicit Catalog(std::span<const CustomTag> custom);
        Catalog(const Catalog&) = delete;
        Catalog& operator=(const Catalog&) = delete;
        std::expected<std::uint32_t, std::string> resolve(std::string_view name) const;
    };
    std::string normalize(std::string_view text);
}
