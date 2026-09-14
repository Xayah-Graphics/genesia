export module genesia.models.registry;
import genesia.io.files;
import std;
export namespace genesia::models {
    struct Descriptor final {
        std::string id, sha;
        std::filesystem::path path;
        std::string fingerprint;
        int step{};
        bool operator==(const Descriptor& other) const {
            return id == other.id && sha == other.sha;
        }
    };
    std::optional<Descriptor> find(std::string_view concept_key);
    Descriptor resolve(std::string_view concept_key);
    Descriptor publish(std::string_view concept_key, const std::filesystem::path& file, std::string fingerprint, int step);
    void unpublish(std::string_view concept_key);
} // namespace genesia::models
