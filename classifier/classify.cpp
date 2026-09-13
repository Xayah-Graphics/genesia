module;
#include "kernels.h"
module classifier.classify;
import classifier.dataset;
import classifier.inference;
import std;
namespace classifier {
    namespace {
        struct Decision {
            std::filesystem::path source;
            std::string label;
            std::vector<std::string> classes;
            std::vector<float> scores;
        };
        std::filesystem::path utf8_path(std::string_view text) {
            std::u8string value;
            value.reserve(text.size());
            for (unsigned char c : text) {
                value.push_back(static_cast<char8_t>(c));
            }
            return std::filesystem::path(value);
        }
        bool is_image_path(const std::filesystem::path& path) {
            auto extension = path_utf8(path.extension());
            std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            return extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp" || extension == ".tga";
        }
        std::vector<std::filesystem::path> direct_image_paths(const std::filesystem::path& folder) {
            std::vector<std::filesystem::path> paths;
            for (const auto& entry : std::filesystem::directory_iterator(folder)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                if (!is_image_path(entry.path())) {
                    continue;
                }
                paths.push_back(entry.path());
            }
            std::ranges::sort(paths);
            return paths;
        }
        std::string score_filename(const std::vector<std::string>& classes, const std::vector<float>& scores) {
            if (classes.size() != scores.size()) {
                throw std::runtime_error("Classifier returned mismatched classes and scores");
            }
            std::vector<std::size_t> order(classes.size());
            std::iota(order.begin(), order.end(), std::size_t{0});
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
            std::string result;
            for (std::size_t index : order) {
                result += std::format("[{}]{:.2f}", classes[index], scores[index]);
            }
            return result;
        }
        std::filesystem::path choose_destination(const std::filesystem::path& directory, const std::string& stem, const std::filesystem::path& extension, std::set<std::filesystem::path>& reserved) {
            for (std::size_t index = 1;; ++index) {
                const std::string name = index == 1 ? stem : std::format("{}-{}", stem, index);
                const auto destination = directory / utf8_path(name + path_utf8(extension));
                if (reserved.contains(destination)) {
                    continue;
                }
                if (std::filesystem::exists(destination)) {
                    continue;
                }
                reserved.insert(destination);
                return destination;
            }
        }
        void move_file(const std::filesystem::path& source, const std::filesystem::path& destination) {
            std::error_code error;
            std::filesystem::rename(source, destination, error);
            if (error) {
                throw std::runtime_error(std::format("Cannot move {} -> {}: {}", path_utf8(source), path_utf8(destination), error.message()));
            }
        }
    } // namespace
    void classify(const ClassifyOptions& options, const std::atomic_bool& interrupted) {
        const auto dataset    = std::filesystem::absolute(options.dataset).lexically_normal();
        const auto input      = std::filesystem::absolute(options.input).lexically_normal();
        const auto model_path = std::filesystem::absolute(options.model.empty() ? dataset / "model.safetensors" : options.model).lexically_normal();
        if (!std::filesystem::exists(input)) {
            throw std::runtime_error("Classification input does not exist: " + path_utf8(input));
        }
        if (!std::filesystem::is_directory(input)) {
            throw std::runtime_error("Classification input is not a directory: " + path_utf8(input));
        }
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("Classifier model does not exist: " + path_utf8(model_path));
        }
        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::runtime_error("Classifier model is not a regular file: " + path_utf8(model_path));
        }
        const auto paths = direct_image_paths(input);
        if (paths.empty()) {
            throw std::runtime_error("No supported images found directly inside "
                                     "classification input: "
                                     + path_utf8(input));
        }
        std::println("Classification: {} images", paths.size());
        std::println("Model: {}", path_utf8(model_path));
        std::println("Directory: {}", path_utf8(input));
        Pipeline pipeline;
        const std::vector<Descriptor> catalog{{"classify", model_path}};
        const Selection selection{{"classify"}};
        std::vector<Decision> decisions;
        decisions.reserve(paths.size());
        for (std::size_t i = 0; i < paths.size(); ++i) {
            if (interrupted.load(std::memory_order_relaxed)) {
                std::println("Interrupted after scoring {}/{}. "
                             "No files were moved.",
                    decisions.size(), paths.size());
                return;
            }
            const auto& path = paths[i];
            try {
                const auto image = load_image(path);
                pipeline.prepare(catalog, selection, image.width, image.height);
                if (pipeline.models.size() != 1) {
                    throw std::runtime_error("Classification pipeline did not load "
                                             "exactly one classifier");
                }
                auto& loaded       = pipeline.models.front();
                const auto shape   = std::pair{image.width, image.height};
                const auto plan_it = loaded.plans.find(shape);
                if (plan_it == loaded.plans.end()) {
                    throw std::runtime_error("Classification pipeline did not prepare "
                                             "the required image shape");
                }
                auto& plan = *plan_it->second;
                cuda_check(cudaMemcpyAsync(plan.pixels.data, image.rgb.data(), image.rgb.size(), cudaMemcpyHostToDevice, plan.stream));
                auto result = pipeline.run({static_cast<unsigned char*>(plan.pixels.data), image.width, image.height, std::size_t(image.width) * 3}, plan.stream);
                if (!result.error.empty()) {
                    throw std::runtime_error(result.error);
                }
                if (result.classifiers.size() != 1) {
                    throw std::runtime_error("Classification pipeline returned "
                                             "an unexpected number of classifier results");
                }
                const auto& classification = result.classifiers.front();
                if (!classification.error.empty()) {
                    throw std::runtime_error(classification.error);
                }
                if (classification.label.empty()) {
                    throw std::runtime_error("Classifier returned an empty label");
                }
                if (classification.classes.size() != classification.scores.size()) {
                    throw std::runtime_error("Classifier returned mismatched classes and scores");
                }
                decisions.push_back({path, classification.label, classification.classes, classification.scores});
            } catch (const std::exception& failure) {
                throw std::runtime_error(std::format("{}: {}", path_utf8(path), failure.what()));
            }
            if ((i + 1) % 100 == 0 || i + 1 == paths.size()) {
                std::println("Scored {}/{}", i + 1, paths.size());
                std::fflush(nullptr);
            }
        }
        if (interrupted.load(std::memory_order_relaxed)) {
            std::println("Interrupted after scoring {}/{}. "
                         "No files were moved.",
                decisions.size(), paths.size());
            return;
        }
        if (pipeline.models.empty()) {
            throw std::runtime_error("Classification pipeline contains no loaded model");
        }
        const auto& classes   = pipeline.models.front().network->classes;
        const float threshold = pipeline.models.front().threshold;
        std::println("Scoring complete. Model threshold: {:.6f}", threshold);
        std::vector<std::filesystem::path> destinations;
        destinations.reserve(decisions.size());
        std::set<std::filesystem::path> reserved;
        for (const auto& decision : decisions) {
            const auto directory   = input / utf8_path(decision.label);
            const auto stem        = score_filename(decision.classes, decision.scores);
            const auto destination = choose_destination(directory, stem, decision.source.extension(), reserved);
            destinations.push_back(destination);
        }
        for (const auto& name : classes) {
            std::filesystem::create_directories(input / utf8_path(name));
        }
        std::map<std::string, std::size_t> counts;
        for (std::size_t i = 0; i < decisions.size(); ++i) {
            const auto& decision    = decisions[i];
            const auto& destination = destinations[i];
            move_file(decision.source, destination);
            ++counts[decision.label];
            if ((i + 1) % 100 == 0 || i + 1 == decisions.size()) {
                std::println("Moved {}/{}", i + 1, decisions.size());
                std::fflush(nullptr);
            }
        }
        std::println("Classified and moved {} images in {}", decisions.size(), path_utf8(input));
        for (const auto& name : classes) {
            std::println("  {}: {}", name, counts[name]);
        }
    }
} // namespace classifier
