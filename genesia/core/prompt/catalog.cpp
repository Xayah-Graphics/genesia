module;
#include <nlohmann/json.hpp>

module genesia.prompt.catalog;
import genesia.generation.defaults;
import std;

namespace genesia::prompt {
    Catalog::Catalog() {
        std::ifstream file{std::filesystem::path{defaults::assets} / "tags/danbooru.csv", std::ios::binary};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        const std::string csv{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
        struct Slice {
            std::uint32_t offset, size;
        };
        struct Row {
            Slice name, text;
            std::uint32_t aliases, alias_count, count;
            int category;
        };
        std::vector<Row> rows;
        std::vector<Slice> alias_offsets;
        const auto append = [this](const std::string_view text) {
            const Slice slice{static_cast<std::uint32_t>(storage.size()), static_cast<std::uint32_t>(text.size())};
            storage.append(text);
            storage += '\0';
            return slice;
        };
        // Offsets are turned into views only after the string arena is complete.
        storage.reserve(csv.size() * 2);
        for (std::size_t cursor = 0; cursor < csv.size();) {
            std::array<std::string, 4> fields;
            for (auto& field : fields) {
                bool quoted = cursor < csv.size() && csv[cursor] == '"';
                if (quoted) ++cursor;
                while (cursor < csv.size()) {
                    const char c = csv[cursor++];
                    if (quoted && c == '"') {
                        if (cursor < csv.size() && csv[cursor] == '"') ++cursor;
                        else {
                            quoted = false;
                            continue;
                        }
                    } else if (!quoted && (c == ',' || c == '\n' || c == '\r')) {
                        if (c == '\r' && cursor < csv.size() && csv[cursor] == '\n') ++cursor;
                        break;
                    }
                    field += c;
                }
            }
            std::string display = fields[0];
            std::ranges::replace(display, '_', ' ');
            Row row{append(fields[0]), append(display), static_cast<std::uint32_t>(alias_offsets.size()), 0, 0, 0};
            std::from_chars(fields[1].data(), fields[1].data() + fields[1].size(), row.category);
            std::from_chars(fields[2].data(), fields[2].data() + fields[2].size(), row.count);
            for (const auto part : std::views::split(fields[3], ',')) {
                const std::string_view alias{part.begin(), part.end()};
                if (alias.empty() || std::ranges::any_of(alias, [](const unsigned char c) { return c >= 128; })) continue;
                alias_offsets.push_back(append(normalize(alias)));
                ++row.alias_count;
            }
            rows.push_back(row);
        }
        std::ifstream custom_file{std::filesystem::path{defaults::assets} / "tags/custom.json"};
        custom_file.exceptions(std::ios::badbit | std::ios::failbit);
        const auto custom = nlohmann::json::parse(custom_file);
        for (const auto& [key, value] : custom.items()) {
            const auto text = value.get<std::string>();
            const auto name = normalize(key);
            if (name.empty() || name.find_first_of(",\r\n") != std::string::npos || std::ranges::any_of(key + text, [](const unsigned char c) { return c >= 128; })) throw std::invalid_argument{"Custom tags require an ASCII name and text; names cannot contain separators"};
            rows.push_back({append(name), append(text), static_cast<std::uint32_t>(alias_offsets.size()), 0, 0, -1});
        }
        const std::string_view arena{storage};
        tags.reserve(rows.size());
        for (const auto& row : rows) {
            const auto id = static_cast<std::uint32_t>(tags.size());
            tags.push_back({arena.substr(row.name.offset, row.name.size), arena.substr(row.text.offset, row.text.size), row.count, row.category});
            names.push_back({tags.back().name, id});
            for (const auto slice : std::span{alias_offsets}.subspan(row.aliases, row.alias_count)) alias_names.push_back({arena.substr(slice.offset, slice.size), id});
        }
        std::ranges::sort(names, {}, &CatalogKey::name);
        std::ranges::sort(alias_names, [](const CatalogKey a, const CatalogKey b) { return std::tie(a.name, a.tag) < std::tie(b.name, b.tag); });
        for (std::size_t i = 1; i < names.size(); ++i)
            if (names[i - 1].name == names[i].name) throw std::invalid_argument{std::format("Duplicate tag: {}", names[i].name)};
        for (const auto& key : names)
            if (tags[key.tag].category == -1 && std::ranges::binary_search(alias_names, key.name, {}, &CatalogKey::name)) throw std::invalid_argument{std::format("Custom tag conflicts with a Danbooru alias: {}", key.name)};
    }

    std::expected<std::uint32_t, std::string> Catalog::resolve(const std::string_view name) const {
        const auto key   = normalize(name);
        const auto found = std::ranges::lower_bound(names, key, {}, &CatalogKey::name);
        if (found != names.end() && found->name == key) return found->tag;
        const auto matches = std::ranges::equal_range(alias_names, key, {}, &CatalogKey::name);
        if (matches.empty()) return std::unexpected{std::format("Unknown tag: {}", name)};
        for (const auto match : matches)
            if (match.tag != matches.front().tag) return std::unexpected{std::format("Ambiguous alias: {}. Choose a suggestion.", name)};
        return matches.front().tag;
    }

    std::string normalize(std::string_view text) {
        const auto begin = text.find_first_not_of(" \t\r\n");
        if (begin == std::string_view::npos) return {};
        text = text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
        std::string result;
        for (char c : text) {
            if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                if (!result.empty() && result.back() != '_') result += '_';
            } else result += c;
        }
        return result;
    }
} // namespace genesia::prompt
