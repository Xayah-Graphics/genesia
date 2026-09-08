module genesia.prompt;
import std;

namespace genesia::prompt {
    std::expected<Input, Error> parse_tag(const Catalog& catalog, std::string_view text, const bool completion) {
        const auto begin = text.find_first_not_of(" \t\r\n");
        if (begin == std::string_view::npos) return std::unexpected{Error{"Enter a tag", 0, text.size()}};
        text                = text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
        const auto unescape = [](const std::string_view source) {
            std::string result;
            for (std::size_t i = 0; i < source.size(); ++i) {
                if (source[i] == '\\' && i + 1 < source.size() && (source[i + 1] == '(' || source[i + 1] == ')')) ++i;
                result += source[i];
            }
            return result;
        };
        // Literal catalog names win, including emoticons and names ending in :1999).
        if (const auto id = catalog.resolve(text)) return Input{std::string{catalog.tags[*id].name}, 1};
        Input result{unescape(text), 1};
        if (const auto id = catalog.resolve(result.name)) return Input{std::string{catalog.tags[*id].name}, 1};
        if (text.front() == '(') {
            if (!completion && text.back() != ')') return std::unexpected{Error{"Close the weighted tag with ')': (tag:1.1)", begin, text.size()}};
            auto inner    = text.substr(1, text.size() - (text.back() == ')' ? 2 : 1));
            result.weight = 1.1F;
            if (const auto id = catalog.resolve(unescape(inner))) return Input{std::string{catalog.tags[*id].name}, result.weight};
            const auto colon = inner.rfind(':');
            if (colon != std::string_view::npos) {
                auto number = inner.substr(colon + 1);
                if (number.starts_with('+')) number.remove_prefix(1);
                const auto parsed = std::from_chars(number.data(), number.data() + number.size(), result.weight);
                if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() || !std::isfinite(result.weight)) return std::unexpected{Error{"Use a finite numeric weight: (tag:1.1)", begin + colon + 2, number.size()}};
                inner = inner.substr(0, colon);
            }
            result.name = unescape(inner);
        }
        return result;
    }

    std::expected<std::vector<Tag>, Error> parse(const Catalog& catalog, const std::string_view text) {
        std::vector<Tag> result;
        for (std::size_t begin = 0; begin < text.size();) {
            const auto end  = std::min(text.find_first_of(",\r\n", begin), text.size());
            const auto part = text.substr(begin, end - begin);
            if (part.find_first_not_of(" \t") != std::string_view::npos) {
                auto input = parse_tag(catalog, part);
                if (!input) {
                    input.error().offset += begin;
                    return std::unexpected{std::move(input.error())};
                }
                const auto id = catalog.resolve(input->name);
                if (!id) return std::unexpected{Error{id.error(), begin, end - begin}};
                result.push_back({*id, input->weight});
            }
            begin = end + 1;
        }
        return result;
    }

    std::string serialize(const Catalog& catalog, const std::span<const Tag> tags, const bool conditioning) {
        std::string result;
        for (const auto tag : tags) {
            if (!std::isfinite(tag.weight)) throw std::invalid_argument{"Tag weight must be finite"};
            if (!result.empty()) result += ", ";
            if (tag.weight != 1) result += '(';
            const auto& entry = catalog.tags[tag.id];
            for (const char c : conditioning ? entry.text : entry.name) {
                if (c == '(' || c == ')') result += '\\';
                result += c;
            }
            if (tag.weight != 1) result += std::format(":{})", tag.weight);
        }
        return result;
    }

    std::string compose(const Catalog& catalog, const Side& side) {
        std::string result;
        for (const auto& group : side.groups) {
            if (!group.enabled || group.tags.empty()) continue;
            if (!result.empty()) result += ", ";
            result += serialize(catalog, group.tags, true);
        }
        if (!result.empty() && !side.fixed.empty()) result += ", ";
        result += side.fixed;
        return result;
    }
} // namespace genesia::prompt
