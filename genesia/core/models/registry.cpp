module;
#include <nlohmann/json.hpp>
module genesia.models.registry;
import genesia.data.concepts;
import genesia.io.files;
import genesia.models.sdxl.lora;
import std;
namespace genesia::models {
    std::optional<Descriptor> find(const std::string_view concept_key) {
        const auto source = dataset::read_concept(concept_key);
        if (source.type == dataset::ConceptType::none) return {};
        const auto root = source.path / ".genesia";
        if (!std::filesystem::exists(root / "model.json")) return {};
        const auto entry = files::read_json(root / "model.json");
        if (entry.at("version") != 1) throw std::runtime_error{"Unsupported model registry format"};
        Descriptor result{.id = source.key, .sha = entry.at("sha").get<std::string>(), .name = source.type == dataset::ConceptType::lora ? entry.at("name").get<std::string>() : files::utf8(source.path.filename()), .type = source.type};
        result.path = root / "models" / (result.sha + ".safetensors");
        return result;
    }
    Descriptor resolve(const std::string_view concept_key, const dataset::ConceptType type) {
        auto result = find(concept_key);
        if (!result || result->type != type) throw std::runtime_error{"Concept has no available " + std::string{dataset::concept_types[std::size_t(type)]} + " model: " + std::string{concept_key}};
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
        return resolve(concept_key, dataset::ConceptType::classifier);
    }
    Descriptor import_lora(const std::string_view concept_key, const std::filesystem::path& file) {
        auto concept_data = dataset::read_concept(concept_key);
        if (concept_data.type != dataset::ConceptType::lora) throw std::runtime_error{"Assign the LoRA type before importing a model: " + concept_data.key};
        const auto previous  = find(concept_key);
        const auto root      = concept_data.path / ".genesia";
        const auto directory = root / "models";
        std::filesystem::create_directories(directory);
        const auto temporary = directory / "import.part";
        std::filesystem::path published;
        bool created{}, locked{};
        try {
            std::filesystem::copy_file(file, temporary, std::filesystem::copy_options::overwrite_existing);
            {
                const files::SafeFile checkpoint{project::checkpoint};
                const sdxl::Adapter adapter{temporary, checkpoint};
            }
            const auto sha = files::digest(temporary);
            published      = directory / (sha + ".safetensors");
            if (std::filesystem::exists(published)) std::filesystem::remove(temporary);
            else {
                files::move(temporary, published);
                created = true;
            }
            if (!concept_data.locked) {
                concept_data.locked = true;
                files::write_json(root / "concept.json", concept_data);
                locked = true;
            }
            files::write_json(root / "model.json", {{"version", 1}, {"sha", sha}, {"name", files::utf8(file.filename())}});
        } catch (...) {
            if (locked) {
                concept_data.locked = false;
                files::write_json(root / "concept.json", concept_data);
            }
            std::filesystem::remove(temporary);
            if (created) std::filesystem::remove(published);
            if (std::filesystem::is_empty(directory)) std::filesystem::remove(directory);
            throw;
        }
        if (previous && previous->path != published) std::filesystem::remove(previous->path);
        return resolve(concept_key, dataset::ConceptType::lora);
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
