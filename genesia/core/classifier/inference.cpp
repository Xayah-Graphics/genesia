module;
#include "kernels.h"
#include <nlohmann/json.hpp>
module genesia.classifier.inference;
import genesia.files;
import genesia.images;
import genesia.classifier.dataset;
import std;
namespace genesia::classifier {
    void to_json(nlohmann::json& json, const Result& result) {
        json = {{"id", result.id}, {"model_sha", result.model_sha}, {"image_sha", result.image_sha}, {"label", result.label}, {"classes", result.classes}, {"scores", result.scores}};
    }
    void from_json(const nlohmann::json& json, Result& result) {
        json.at("id").get_to(result.id);
        json.at("model_sha").get_to(result.model_sha);
        json.at("image_sha").get_to(result.image_sha);
        json.at("label").get_to(result.label);
        json.at("classes").get_to(result.classes);
        json.at("scores").get_to(result.scores);
    }
    Descriptor model(const std::string_view key) {
        const auto source = dataset::read_concept(key);
        const auto root   = source.path / ".genesia" / "classifier";
        const auto state  = read_training(source);
        if (state.is_null()) throw std::runtime_error{"Classifier has no training record: " + source.key};
        const auto& published = state.at("published");
        if (published.is_null()) throw std::runtime_error{"Classifier training has not completed: " + std::string(key)};
        return {source.key, published.at("sha"), root / "models" / (published.at("sha").get<std::string>() + ".safetensors")};
    }
    Pipeline::~Pipeline() {
        if (pending_stream) cudaStreamSynchronize(pending_stream);
        for (auto& model : models) model.plans.clear();
        arena.reset();
        models.clear();
        if (host_results) cudaFreeHost(host_results);
    }
    void Pipeline::prepare(std::vector<Descriptor> desired, int width, int height) {
        std::ranges::sort(desired, {}, &Descriptor::id);
        if (!desired.empty() && !((width == 1024 && height == 1536) || (width == 1536 && height == 1024) || (width == 1024 && height == 1024))) throw std::runtime_error("Enabled classifiers require 1024x1536, 1536x1024, or 1024x1024 original RGB8 input");
        const bool changed = desired.size() != models.size() || !std::equal(desired.begin(), desired.end(), models.begin(), [](const Descriptor& a, const LoadedClassifier& b) { return a == b.descriptor; });
        if (changed) {
            if (pending_stream) cuda_check(cudaStreamSynchronize(pending_stream));
            for (auto& model : models) model.plans.clear();
            arena.reset();
            std::vector<LoadedClassifier> next;
            for (const auto& descriptor : desired) {
                auto old = std::ranges::find_if(models, [&](const LoadedClassifier& m) { return m.descriptor == descriptor; });
                if (old != models.end()) next.push_back(std::move(*old));
                else {
                    LoadedClassifier entry;
                    entry.descriptor = descriptor;
                    if (files::digest(descriptor.path) != descriptor.sha) throw std::runtime_error{"Classifier model does not match its published SHA"};
                    entry.network = std::make_unique<Network>(descriptor.path);
                    next.push_back(std::move(entry));
                }
            }
            models       = std::move(next);
            result_bytes = 0;
            for (auto& m : models) {
                m.offset = result_bytes;
                result_bytes += (m.network->classes.size() * 4 + 4 + 255) / 256 * 256;
            }
            result_storage = DeviceBuffer(result_bytes);
            if (host_results) cudaFreeHost(std::exchange(host_results, nullptr));
            if (result_bytes) cuda_check(cudaMallocHost(&host_results, result_bytes));
            for (auto& m : models) {
                auto base        = static_cast<std::byte*>(result_storage.data) + m.offset;
                const auto count = m.network->classes.size();
                m.output         = {reinterpret_cast<float*>(base), reinterpret_cast<int*>(base + count * 4)};
            }
            if (!models.empty()) {
                const auto largest = std::ranges::max_element(models, {}, [](const LoadedClassifier& m) { return m.network->classes.size(); });
                arena              = std::make_unique<NetworkPlan>(*largest->network, 1, 1024, 1536);
            }
        }
        for (auto& m : models) {
            const auto shape = std::pair{width, height};
            if (!m.plans.contains(shape)) {
                m.plans[shape] = std::make_unique<NetworkPlan>(*m.network, 1, width, height, ActivationStorage::reuse, arena.get());
                m.plans[shape]->capture();
            }
        }
    }
    Result Pipeline::run(const Descriptor& descriptor, GpuRgb8 input, cudaStream_t stream, const std::string_view image_sha) {
        pending_stream = stream;
        auto& m        = *std::ranges::find(models, descriptor, &LoadedClassifier::descriptor);
        m.plans.at({input.width, input.height})->infer(input, m.output, stream);
        const auto count = m.network->classes.size();
        cuda_check(cudaMemcpyAsync(host_results + m.offset, static_cast<std::byte*>(result_storage.data) + m.offset, count * 4 + 4, cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        const auto base   = host_results + m.offset;
        const auto scores = reinterpret_cast<const float*>(base);
        const int index   = *reinterpret_cast<const int*>(base + count * 4);
        return {m.descriptor.id, m.descriptor.sha, std::string(image_sha), m.network->classes[index], m.network->classes, {scores, scores + count}};
    }
    std::optional<Result> Predictions::find(const Descriptor& descriptor, const std::string_view image_sha) {
        if (!cached.contains(descriptor.sha)) {
            const auto path        = dataset::state_directory / "inference" / (descriptor.sha + ".json");
            cached[descriptor.sha] = std::filesystem::exists(path) ? files::read_json(path) : nlohmann::json::object();
        }
        const auto& entries = cached.at(descriptor.sha);
        if (!entries.contains(image_sha)) return {};
        auto result = entries.at(image_sha).get<Result>();
        result.id   = descriptor.id;
        return result;
    }
    Result Predictions::infer(const Descriptor& descriptor, const dataset::File& file, const bool refresh) {
        if (!refresh)
            if (auto result = find(descriptor, file.sha)) return *result;
        std::vector<Descriptor> desired;
        for (const auto& loaded : pipeline.models)
            if (loaded.descriptor.id != descriptor.id) desired.push_back(loaded.descriptor);
        desired.push_back(descriptor);
        pipeline.prepare(std::move(desired), file.width, file.height);
        auto& loaded     = *std::ranges::find(pipeline.models, descriptor, &LoadedClassifier::descriptor);
        auto& plan       = *loaded.plans.at({file.width, file.height});
        const auto image = read_image(file.path);
        if (files::digest(file.path) != file.sha) throw std::runtime_error{"Image changed before inference: " + files::utf8(file.path)};
        cuda_check(cudaMemcpyAsync(plan.pixels.data, image.pixels.data(), image.pixels.size(), cudaMemcpyHostToDevice, plan.stream));
        auto result = pipeline.run(descriptor, {static_cast<unsigned char*>(plan.pixels.data), file.width, file.height, std::size_t(file.width) * 3}, plan.stream, file.sha);
        const dataset::Lock lock{"inference-" + descriptor.sha};
        const auto path  = dataset::state_directory / "inference" / (descriptor.sha + ".json");
        auto stored      = std::filesystem::exists(path) ? files::read_json(path) : nlohmann::json::object();
        stored[file.sha] = result;
        files::write_json(path, stored);
        cached[descriptor.sha] = std::move(stored);
        return result;
    }
} // namespace genesia::classifier
