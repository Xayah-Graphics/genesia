module;
#include <nlohmann/json.hpp>
module genesia.classifier.classify;
import genesia.classifier.dataset;
import genesia.dataset.operations;
import genesia.files;
import genesia.hash;
import std;
namespace genesia::classifier {
    nlohmann::json classify(const std::string_view key, const std::filesystem::path& input, Predictions& predictions, const std::atomic_bool& interrupted, const std::function<void(const nlohmann::json&)>& progress, const std::function<void()>& yield) {
        const auto root       = std::filesystem::absolute(input).lexically_normal();
        const auto text       = files::utf8(root);
        const auto folder_sha = sha256({reinterpret_cast<const unsigned char*>(text.data()), text.size()});
        const dataset::Lock lock{"classify-" + folder_sha};
        const auto journal = dataset::state_directory / "operations" / (folder_sha + ".json");
        {
            const dataset::Lock recovery{"image-moves"};
            dataset::recover_moves(journal);
        }
        dataset::Index index;
        const auto relative = root.lexically_relative(dataset::directory);
        std::unique_ptr<dataset::Lock> concept_lock;
        if (!relative.empty() && *relative.begin() != ".." && std::distance(relative.begin(), relative.end()) >= 2) {
            auto part               = relative.begin();
            const auto dataset_name = *part++;
            const auto concept_key  = files::utf8(dataset_name / *part);
            concept_lock            = std::make_unique<dataset::Lock>("concept-" + sha256({reinterpret_cast<const unsigned char*>(concept_key.data()), concept_key.size()}));
        }
        if (!relative.empty() && *relative.begin() != "..") {
            index.scan(files::utf8(*relative.begin()));
            if (!index.roots.front().ready) throw std::runtime_error{"Dataset root is not ready"};
        }
        std::vector<dataset::File> images;
        std::map<std::string, std::string> entities;
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (!entry.is_regular_file()) continue;
            auto extension = files::utf8(entry.path().extension());
            std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension != ".png") continue;
            auto image                 = index.identify(entry.path());
            const auto [entity, added] = entities.emplace(image.sha, image.entity);
            if (!added && entity->second != image.entity) throw std::runtime_error{"Independent copies in classification input: " + files::utf8(image.path)};
            images.push_back(std::move(image));
        }
        if (images.empty()) throw std::runtime_error{"No PNG images directly inside " + text};
        std::ranges::sort(images, {}, &dataset::File::path);
        const auto descriptor = model(key);
        std::vector<dataset::Move> moves;
        nlohmann::json counts = nlohmann::json::object();
        for (const auto& image : images) {
            if (interrupted.load()) throw std::runtime_error{"Classification stopped before moving files"};
            const auto result = predictions.infer(descriptor, image);
            moves.push_back({image.path, root / files::path(result.label) / image.path.filename(), image.sha, image.entity});
            counts[result.label] = counts.value(result.label, 0) + 1;
            progress({{"stage", "classifying"}, {"completed", moves.size()}, {"total", images.size()}});
            yield();
        }
        progress({{"stage", "moving"}, {"completed", images.size()}, {"total", images.size()}});
        if (interrupted.load()) throw std::runtime_error{"Classification stopped before moving files"};
        auto result       = dataset::move_images(std::move(moves), journal);
        result["classes"] = counts;
        result["input"]   = text;
        return result;
    }
} // namespace genesia::classifier
