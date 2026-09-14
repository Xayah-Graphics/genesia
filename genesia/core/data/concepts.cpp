module;
#include <nlohmann/json.hpp>
module genesia.data.concepts;
import genesia.project;
import genesia.io.files;
import std;
namespace genesia::dataset {
    ConceptType parse_concept_type(const std::string_view name) {
        const auto found = std::ranges::find(concept_types, name);
        if (found == concept_types.end()) throw std::runtime_error{"Unknown concept type: " + std::string(name)};
        return static_cast<ConceptType>(found - concept_types.begin());
    }
    void to_json(nlohmann::json& json, const Concept& value) {
        json = {{"version", 2}, {"type", concept_types[static_cast<std::size_t>(value.type)]}, {"locked", value.locked}};
    }
    Concept read_concept(const std::string_view key) {
        const auto relative = files::path(key);
        if (relative.is_absolute() || std::distance(relative.begin(), relative.end()) != 2 || std::ranges::any_of(relative, [](const auto& part) { return files::utf8(part).starts_with('.'); })) throw std::runtime_error{"Expected ROOT/CONCEPT"};
        Concept result{files::utf8(relative), project::directory / relative};
        if (!std::filesystem::is_directory(result.path)) throw std::runtime_error{"Concept not found: " + result.key};
        const auto state    = result.path / ".genesia";
        const auto manifest = state / "concept.json";
        if (std::filesystem::exists(manifest)) {
            const auto value = files::read_json(manifest);
            if (value.at("version") != 2) throw std::runtime_error{"Old or unsupported concept state: " + result.key + ". Remove the old training artifacts before assigning a type."};
            result.type   = parse_concept_type(value.at("type").get_ref<const std::string&>());
            result.locked = value.at("locked").get<bool>();
            if (result.type == ConceptType::none && result.locked) throw std::runtime_error{"An unassigned concept cannot have a locked training type: " + result.key};
        } else if (std::filesystem::exists(state) && !std::filesystem::is_empty(state)) throw std::runtime_error{"Concept state has no type manifest: " + result.key};
        return result;
    }
    Concept assign_type(const std::string_view key, const ConceptType type) {
        const auto normalized = files::utf8(files::path(key));
        auto result           = read_concept(normalized);
        if (result.locked) throw std::runtime_error{"Concept type is locked by its training history: " + result.key};
        result.type = type;
        files::write_json(result.path / ".genesia" / "concept.json", result);
        return result;
    }

} // namespace genesia::dataset
