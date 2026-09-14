module;
#include <nlohmann/json.hpp>
module genesia.models.registry;
import genesia.data.concepts;
import genesia.io.files;
import genesia.io.hash;
import std;
namespace genesia::models {
    std::optional<Descriptor> find(const std::string_view concept_key) {
        const auto source = dataset::read_concept(concept_key);
        if (source.type != dataset::ConceptType::classifier) return {};
        collect(concept_key);
        const auto root = source.path / ".genesia";
        const files::Lock lock{"registry-" + sha256({reinterpret_cast<const unsigned char*>(source.key.data()), source.key.size()})};
        if (!std::filesystem::exists(root / "model.json")) return {};
        const auto entry = files::read_json(root / "model.json");
        if (entry.at("version") != 1) throw std::runtime_error{"Unsupported model registry format"};
        Descriptor result{source.key, entry.at("sha").get<std::string>()};
        result.path        = root / "models" / (result.sha + ".safetensors");
        result.fingerprint = entry.at("fingerprint");
        result.step        = entry.at("step");
        result.lease       = std::make_shared<files::Lock>("model-" + result.sha, true, project::state_directory, true);
        return result;
    }
    Descriptor resolve(const std::string_view concept_key) {
        auto result = find(concept_key);
        if (!result) throw std::runtime_error{"Concept has no published classifier: " + std::string{concept_key}};
        return std::move(*result);
    }
    Descriptor publish(const std::string_view concept_key, const std::filesystem::path& file, std::string fingerprint, const int step) {
        const auto source = dataset::read_concept(concept_key);
        const auto root   = source.path / ".genesia";
        const auto sha    = files::digest(file);
        {
            const files::Lock lock{"registry-" + sha256({reinterpret_cast<const unsigned char*>(source.key.data()), source.key.size()})};
            const auto path = root / "models" / (sha + ".safetensors");
            std::filesystem::create_directories(path.parent_path());
            if (std::filesystem::exists(path)) std::filesystem::remove(file);
            else files::move(file, path);
            files::write_json(root / "model.json", {{"version", 1}, {"sha", sha}, {"fingerprint", fingerprint}, {"step", step}});
        }
        collect(concept_key);
        return resolve(concept_key);
    }
    void unpublish(const std::string_view concept_key) {
        const auto source = dataset::read_concept(concept_key);
        {
            const files::Lock lock{"registry-" + sha256({reinterpret_cast<const unsigned char*>(source.key.data()), source.key.size()})};
            std::filesystem::remove(source.path / ".genesia" / "model.json");
        }
        collect(concept_key);
    }
    void collect(const std::string_view concept_key) {
        const auto source = dataset::read_concept(concept_key);
        const auto root   = source.path / ".genesia";
        const files::Lock lock{"registry-" + sha256({reinterpret_cast<const unsigned char*>(source.key.data()), source.key.size()})};
        const auto current = std::filesystem::exists(root / "model.json") ? files::read_json(root / "model.json").at("sha").get<std::string>() : std::string{};
        if (!std::filesystem::exists(root / "models")) return;
        for (const auto& entry : std::filesystem::directory_iterator{root / "models"}) {
            if (entry.path().extension() != ".safetensors") continue;
            const auto sha = files::utf8(entry.path().stem());
            if (sha == current) continue;
            const files::Lock reader{"model-" + sha, false};
            if (reader.acquired) std::filesystem::remove(entry.path());
        }
    }
} // namespace genesia::models
