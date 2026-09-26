module;
#include <nlohmann/json.hpp>
module classifier.data.models;
import tools.files;
import std;
namespace classifier::models {
    std::optional<Descriptor> find(const std::filesystem::path& root) {
        const auto directory = root / ".genesia";
        if (!std::filesystem::exists(directory)) return {};
        const auto manifest = tools::files::read_json(directory / "concept.json");
        if (manifest.at("version") != 2 || manifest.at("type") != "classifier") throw std::runtime_error{"This directory is not a classifier"};
        if (!std::filesystem::exists(directory / "model.json")) return {};
        const auto entry = tools::files::read_json(directory / "model.json");
        if (entry.at("version") != 1) throw std::runtime_error{"Unsupported model registry format"};
        const auto sha = entry.at("sha").get<std::string>();
        return Descriptor{directory / "models" / (sha + ".safetensors"), sha};
    }
    Descriptor resolve(const std::filesystem::path& root) {
        auto model = find(root);
        if (!model) throw std::runtime_error{"Complete training to publish a classifier model"};
        return std::move(*model);
    }
    Info inspect(const std::filesystem::path& root) {
        Info result{root, find(root)};
        if (!result.model) return result;
        std::ifstream file{result.model->path, std::ios::binary};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::uint64_t length{};
        file.read(reinterpret_cast<char*>(&length), sizeof(length));
        std::string text(length, '\0');
        file.read(text.data(), text.size());
        const auto header    = nlohmann::json::parse(text);
        const auto& metadata = header.at("__metadata__");
        if (metadata.at("format") != "genesia-convnext-2") throw std::runtime_error{"Unsupported classifier model format"};
        result.classes = nlohmann::json::parse(metadata.at("classes").get<std::string>()).get<std::vector<std::string>>();
        result.step    = std::stoi(metadata.at("step").get<std::string>());
        return result;
    }
    void publish(const std::filesystem::path& root, const std::filesystem::path& file, std::string fingerprint, const int step) {
        const auto previous = find(root);
        const auto sha      = tools::files::digest(file);
        const auto path     = root / ".genesia" / "models" / (sha + ".safetensors");
        std::filesystem::create_directories(path.parent_path());
        if (std::filesystem::exists(path)) std::filesystem::remove(file);
        else tools::files::move(file, path);
        tools::files::write_json(root / ".genesia" / "model.json", {{"version", 1}, {"sha", sha}, {"fingerprint", fingerprint}, {"step", step}});
        if (previous && previous->sha != sha) std::filesystem::remove(previous->path);
    }
    void unpublish(const std::filesystem::path& root) {
        const auto previous = find(root);
        std::filesystem::remove(root / ".genesia" / "model.json");
        if (previous) std::filesystem::remove(previous->path);
        const auto directory = root / ".genesia" / "models";
        if (std::filesystem::exists(directory) && std::filesystem::is_empty(directory)) std::filesystem::remove(directory);
    }
} // namespace classifier::models
