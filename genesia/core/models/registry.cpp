module;
#include <nlohmann/json.hpp>
module genesia.models.registry;
import genesia.data.concepts;
import genesia.io.files;
import std;
namespace genesia::models {
    std::optional<Descriptor> find(const std::string_view concept_key) {
        const auto source = dataset::read_concept(concept_key);
        if (source.type != dataset::ConceptType::classifier) return {};
        const auto root = source.path / ".genesia";
        if (!std::filesystem::exists(root / "model.json")) return {};
        const auto entry = files::read_json(root / "model.json");
        if (entry.at("version") != 1) throw std::runtime_error{"Unsupported model registry format"};
        Descriptor result{source.key, entry.at("sha").get<std::string>()};
        result.path        = root / "models" / (result.sha + ".safetensors");
        result.fingerprint = entry.at("fingerprint");
        result.step        = entry.at("step");
        return result;
    }
    Descriptor resolve(const std::string_view concept_key) {
        auto result = find(concept_key);
        if (!result) throw std::runtime_error{"Concept has no published classifier: " + std::string{concept_key}};
        return std::move(*result);
    }
    Descriptor publish(const std::string_view concept_key, const std::filesystem::path& file, std::string fingerprint, const int step) {
        const auto source   = dataset::read_concept(concept_key);
        const auto root     = source.path / ".genesia";
        const auto previous = find(concept_key);
        const auto sha      = files::digest(file);
        {
            const auto path = root / "models" / (sha + ".safetensors");
            std::filesystem::create_directories(path.parent_path());
            if (std::filesystem::exists(path)) std::filesystem::remove(file);
            else files::move(file, path);
            files::write_json(root / "model.json", {{"version", 1}, {"sha", sha}, {"fingerprint", fingerprint}, {"step", step}});
        }
        if (previous && previous->sha != sha) std::filesystem::remove(previous->path);
        return resolve(concept_key);
    }
    void unpublish(const std::string_view concept_key) {
        const auto previous = find(concept_key);
        const auto source   = dataset::read_concept(concept_key);
        {
            std::filesystem::remove(source.path / ".genesia" / "model.json");
            if (previous) std::filesystem::remove(previous->path);
            const auto directory = source.path / ".genesia" / "models";
            if (std::filesystem::exists(directory) && std::filesystem::is_empty(directory)) std::filesystem::remove(directory);
        }
    }
} // namespace genesia::models
