export module genesia.models.registry;
export import genesia.data.concepts;
import genesia.io.files;
import std;
export namespace genesia::models {
    struct Descriptor final {
        std::string id, sha, name;
        std::filesystem::path path;
        dataset::ConceptType type{};
        bool operator==(const Descriptor& other) const {
            return id == other.id && sha == other.sha;
        }
    };
    std::optional<Descriptor> find(std::string_view concept_key);
    Descriptor resolve(std::string_view concept_key, dataset::ConceptType type);
    Descriptor publish(std::string_view concept_key, const std::filesystem::path& file, std::string fingerprint, int step);
    Descriptor import_lora(std::string_view concept_key, const std::filesystem::path& file);
    void unpublish(std::string_view concept_key, bool replacing = false);
} // namespace genesia::models
