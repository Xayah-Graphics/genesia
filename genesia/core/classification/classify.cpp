module genesia.classification.classify;
import genesia.models.registry;
import genesia.project;
import genesia.training.samples;
import genesia.data.operations;
import genesia.io.files;
import std;
namespace genesia::classification {
    Classification classify(dataset::Index& index, const std::string_view key, const std::filesystem::path& input, Predictions& predictions, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress) {
        const auto root     = std::filesystem::absolute(input).lexically_normal();
        const auto text     = files::utf8(root);
        const auto relative = root.lexically_relative(project::directory);
        std::vector<dataset::File> images;
        if (!relative.empty() && *relative.begin() != "..") {
            const auto& dataset = *std::ranges::find(index.roots, files::utf8(*relative.begin()), [](const dataset::Root& root) { return root.all.key; });
            if (!dataset.ready) throw std::runtime_error{"Dataset root is not ready"};
            for (const auto& file : dataset.files)
                if (file.path.parent_path() == root) images.push_back(file);
        } else {
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
        }
        if (images.empty()) throw std::runtime_error{"No PNG images directly inside " + text};
        std::ranges::sort(images, {}, &dataset::File::path);
        const auto descriptor = models::resolve(key, dataset::ConceptType::classifier);
        std::vector<dataset::Move> moves;
        std::map<std::string, std::size_t> counts;
        for (const auto& image : images) {
            if (interrupted.load()) throw runtime::Stopped{};
            const auto result = predictions.infer(descriptor, image);
            moves.push_back({image.path, root / files::path(result.label) / image.path.filename(), image.sha});
            ++counts[result.label];
            progress({runtime::BatchProgress{runtime::Stage::classifying, moves.size(), images.size()}});
        }
        if (!relative.empty() && *relative.begin() != "..") {
            auto changed = *std::ranges::find(index.roots, files::utf8(*relative.begin()), [](const dataset::Root& value) { return value.all.key; });
            std::set<std::string> affected;
            for (const auto& move : moves) {
                const auto file        = std::ranges::find(changed.files, move.source, &dataset::File::path);
                file->path             = move.destination;
                const auto destination = move.destination.lexically_relative(project::directory);
                if (std::distance(destination.begin(), destination.end()) < 3) continue;
                auto part       = destination.begin();
                const auto name = *part++;
                affected.insert(files::utf8(name / *part));
            }
            for (const auto& key : affected) {
                if (!std::filesystem::is_directory(project::directory / files::path(key))) continue;
                if (dataset::read_concept(key).type != dataset::ConceptType::lora) continue;
                const auto issue = dataset::lora_issue(changed, key);
                if (!issue.empty()) throw std::runtime_error{issue};
            }
        }
        progress({runtime::BatchProgress{runtime::Stage::moving, images.size(), images.size()}});
        if (interrupted.load()) throw runtime::Stopped{};
        return {root, dataset::move_images(std::move(moves)), std::move(counts)};
    }
} // namespace genesia::classification
