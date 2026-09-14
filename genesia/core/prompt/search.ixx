export module genesia.prompt.search;
import genesia.prompt.catalog;
import std;
export namespace genesia::prompt {
    struct TagSuggestion final {
        std::uint32_t tag;
        std::string_view match;
        int rank;
    };
    struct TagSearch final {
        struct Posting {
            std::uint32_t gram, begin, count;
        };
        const Catalog& catalog;
        std::vector<CatalogKey> keys;
        std::vector<Posting> index;
        std::vector<std::uint32_t> postings;

        explicit TagSearch(const Catalog& catalog);
        std::vector<TagSuggestion> search(std::string_view query) const;
    };
} // namespace genesia::prompt
