module genesia.prompt.document;
import std;
namespace genesia::prompt {
    Document store(const Pair& prompt, const Catalog& catalog) {
        Document result;
        for (const auto& [source, target] : {std::pair{&prompt.positive, &result.sides[0]}, std::pair{&prompt.negative, &result.sides[1]}}) {
            target->text  = compose(catalog, *source);
            target->fixed = source->fixed;
            for (const auto& group : source->groups) {
                auto& saved = target->groups.emplace_back(group.enabled);
                for (const auto tag : group.tags) {
                    const auto& entry = catalog.tags[tag.id];
                    saved.tags.push_back({std::string{entry.name}, std::string{entry.text}, tag.weight});
                }
            }
        }
        return result;
    }
    Resolved resolve(const Document& document, std::shared_ptr<const Catalog> catalog) {
        std::vector<ArchivedTag> archived;
        for (const auto& side : document.sides)
            for (const auto& group : side.groups)
                for (const auto& tag : group.tags) {
                    const auto id = catalog->resolve(tag.name);
                    if ((!id || catalog->tags[*id].name != tag.name || catalog->tags[*id].text != tag.text) && !std::ranges::contains(archived, tag.name, &ArchivedTag::name)) archived.push_back({tag.name, tag.text});
                }
        if (!archived.empty()) catalog = std::make_shared<Catalog>(std::move(catalog), archived);
        Resolved result{std::move(catalog)};
        for (const auto& [source, target] : {std::pair{&document.sides[0], &result.prompt.positive}, std::pair{&document.sides[1], &result.prompt.negative}}) {
            target->fixed = source->fixed;
            for (const auto& group : source->groups) {
                auto& resolved = target->groups.emplace_back(std::vector<Tag>{}, group.enabled);
                for (const auto& tag : group.tags) resolved.tags.push_back({result.catalog->resolve(tag.name).value(), tag.weight});
            }
        }
        return result;
    }
} // namespace genesia::prompt
