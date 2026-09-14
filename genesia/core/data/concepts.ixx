module;
#include <nlohmann/json.hpp>
export module genesia.data.concepts;
export import genesia.project;
import std;
export namespace genesia::dataset {
    enum class ConceptType { none, classifier, lora };
    inline constexpr std::array<std::string_view, 3> concept_types{"none", "classifier", "lora"};
    struct Concept final {
        std::string key;
        std::filesystem::path path;
        ConceptType type{ConceptType::none};
        bool locked{};
    };
    ConceptType parse_concept_type(std::string_view name);
    void to_json(nlohmann::json& json, const Concept& value);
    Concept read_concept(std::string_view key);
    Concept assign_type(std::string_view key, ConceptType type);

} // namespace genesia::dataset
