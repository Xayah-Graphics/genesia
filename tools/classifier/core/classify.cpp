module classifier.classify;
import std;
namespace classifier::classification {
    Classification classify(const models::Descriptor& model, const std::filesystem::path& input, Predictions& predictions, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress) {
        dataset::Index index{input};
        std::vector<dataset::File> images;
        for (const auto& entry : std::filesystem::directory_iterator(input)) {
            if (interrupted) throw runtime::Stopped{};
            if (!entry.is_regular_file()) continue;
            auto extension = tools::files::utf8(entry.path().extension());
            std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension == ".png") images.push_back(index.identify(entry.path()));
        }
        index.flush();
        if (images.empty()) throw std::runtime_error{"No PNG images directly inside " + tools::files::utf8(input)};
        std::ranges::sort(images, {}, &dataset::File::path);
        Classification result{input};
        std::vector<dataset::Move> moves;
        for (const auto& image : images) {
            if (interrupted) throw runtime::Stopped{};
            const auto prediction = predictions.infer(model, image, false, progress);
            moves.push_back({image.path, input / tools::files::path(prediction.label) / image.path.filename()});
            ++result.classes[prediction.label];
            progress({runtime::BatchProgress{runtime::Stage::classifying, moves.size(), images.size(), tools::files::utf8(image.path)}});
        }
        progress({runtime::BatchProgress{runtime::Stage::moving, images.size(), images.size()}});
        if (interrupted) throw runtime::Stopped{};
        result.movement = dataset::move_images(std::move(moves));
        return result;
    }
} // namespace classifier::classification
