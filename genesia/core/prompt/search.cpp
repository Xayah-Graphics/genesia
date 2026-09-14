module genesia.prompt.search;
import std;
namespace genesia::prompt {
    TagSearch::TagSearch(const Catalog& source) : catalog{source}, keys{source.names} {
        keys.append_range(source.alias_names);
        std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
        std::vector<std::uint32_t> grams;
        for (std::uint32_t id = 0; id < keys.size(); ++id) {
            grams.clear();
            const auto key = keys[id].name;
            for (std::size_t i = 0; i < key.size(); ++i) {
                std::uint32_t gram{};
                for (std::size_t n = 0; n < 3 && i + n < key.size(); ++n) {
                    gram |= std::uint32_t{static_cast<unsigned char>(key[i + n])} << (8 * n);
                    grams.push_back(gram);
                }
            }
            std::ranges::sort(grams);
            grams.erase(std::unique(grams.begin(), grams.end()), grams.end());
            for (const auto gram : grams) pairs.emplace_back(gram, id);
        }
        std::ranges::sort(pairs);
        postings.reserve(pairs.size());
        for (const auto [gram, key] : pairs) {
            if (index.empty() || index.back().gram != gram) index.push_back({gram, static_cast<std::uint32_t>(postings.size()), 0});
            postings.push_back(key);
            ++index.back().count;
        }
    }

    std::vector<TagSuggestion> TagSearch::search(const std::string_view text) const {
        const auto query = normalize(text);
        std::vector<TagSuggestion> result;
        if (query.empty()) return result;
        const std::size_t length = std::min(3uz, query.size());
        const Posting* rarest{};
        for (std::size_t i = 0; i + length <= query.size(); ++i) {
            std::uint32_t gram{};
            for (std::size_t n = 0; n < length; ++n) gram |= std::uint32_t{static_cast<unsigned char>(query[i + n])} << (8 * n);
            const auto found = std::ranges::lower_bound(index, gram, {}, &Posting::gram);
            if (found == index.end() || found->gram != gram) return result;
            if (!rarest || found->count < rarest->count) rarest = &*found;
        }
        const auto better = [this](const TagSuggestion a, const TagSuggestion b) {
            if (a.rank != b.rank) return a.rank < b.rank;
            const auto& x = catalog.tags[a.tag];
            const auto& y = catalog.tags[b.tag];
            return x.count != y.count ? x.count > y.count : x.name < y.name;
        };
        for (const auto id : std::span{postings}.subspan(rarest->begin, rarest->count)) {
            const auto key      = keys[id];
            const auto position = key.name.find(query);
            if (position == std::string_view::npos) continue;
            const bool alias = key.name != catalog.tags[key.tag].name;
            const TagSuggestion candidate{key.tag, key.name, (key.name == query ? 0 : position == 0 ? 2 : 4) + int(alias)};
            const auto duplicate = std::ranges::find(result, key.tag, &TagSuggestion::tag);
            if (duplicate != result.end()) {
                if (!better(candidate, *duplicate)) continue;
                result.erase(duplicate);
            }
            result.insert(std::ranges::lower_bound(result, candidate, better), candidate);
            if (result.size() > 8) result.pop_back();
        }
        return result;
    }

} // namespace genesia::prompt
