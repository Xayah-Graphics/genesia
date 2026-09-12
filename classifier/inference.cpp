module;
#include "kernels.h"

#include <nlohmann/json.hpp>
module classifier.inference;
import std;
namespace classifier {
    std::filesystem::path model_directory() {
        return std::filesystem::path(u8"" CLASSIFIER_MODEL_ROOT);
    }
    void to_json(nlohmann::json& json, const Result& result) {
        json = {{"id", result.id}, {"label", result.label}, {"classes", result.classes}, {"scores", result.scores}, {"threshold", result.threshold}, {"accepted", result.accepted}, {"error", result.error}};
    }
    void from_json(const nlohmann::json& json, Result& result) {
        json.at("id").get_to(result.id);
        json.at("label").get_to(result.label);
        json.at("classes").get_to(result.classes);
        json.at("scores").get_to(result.scores);
        json.at("threshold").get_to(result.threshold);
        json.at("accepted").get_to(result.accepted);
        json.at("error").get_to(result.error);
    }
    void to_json(nlohmann::json& json, const Results& results) {
        json = {{"classifiers", results.classifiers}, {"passed", results.passed}, {"error", results.error}};
    }
    void from_json(const nlohmann::json& json, Results& results) {
        json.at("classifiers").get_to(results.classifiers);
        json.at("passed").get_to(results.passed);
        json.at("error").get_to(results.error);
    }
    std::vector<Descriptor> discover() {
        const auto root = model_directory();
        std::vector<Descriptor> result;
        if (!std::filesystem::exists(root)) return result;
        for (const auto& file : std::filesystem::directory_iterator(root)) {
            if (!file.is_regular_file() || file.path().extension() != ".safetensors") continue;
            result.push_back({path_utf8(file.path().stem()), file.path(), file.last_write_time()});
        }
        std::ranges::sort(result, {}, &Descriptor::id);
        return result;
    }
    Pipeline::~Pipeline() {
        if (pending_stream) cudaStreamSynchronize(pending_stream);
        for (auto& model : models) model.plans.clear();
        arena.reset();
        models.clear();
        if (host_results) cudaFreeHost(host_results);
    }
    void Pipeline::prepare(const std::vector<Descriptor>& catalog, const Selection& selection, int width, int height) {
        std::vector<Descriptor> desired;
        for (const auto& id : selection.enabled) {
            const auto found = std::ranges::find(catalog, id, &Descriptor::id);
            if (found == catalog.end()) throw std::runtime_error("Classifier not found: " + id);
            desired.push_back(*found);
        }
        std::ranges::sort(desired, {}, &Descriptor::id);
        desired.erase(std::unique(desired.begin(), desired.end()), desired.end());
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
                    entry.network    = std::make_unique<Network>(descriptor.path);
                    entry.threshold  = std::stof(entry.network->metadata.value("threshold", "0.9"));
                    next.push_back(std::move(entry));
                }
            }
            models       = std::move(next);
            result_bytes = 0;
            for (auto& m : models) {
                m.offset = result_bytes;
                result_bytes += (m.network->classes.size() * 4 + 8 + 255) / 256 * 256;
            }
            result_storage = DeviceBuffer(result_bytes);
            if (host_results) cudaFreeHost(std::exchange(host_results, nullptr));
            if (result_bytes) cuda_check(cudaMallocHost(&host_results, result_bytes));
            for (auto& m : models) {
                auto base        = static_cast<std::byte*>(result_storage.data) + m.offset;
                const auto count = m.network->classes.size();
                m.output         = {reinterpret_cast<float*>(base), reinterpret_cast<int*>(base + count * 4), reinterpret_cast<unsigned char*>(base + count * 4 + 4)};
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
    Results Pipeline::run(GpuRgb8 input, cudaStream_t stream) {
        Results result;
        if (models.empty()) return result;
        pending_stream = stream;
        try {
            for (auto& m : models) m.plans.at({input.width, input.height})->infer(input, m.output, stream, m.threshold);
            cuda_check(cudaMemcpyAsync(host_results, result_storage.data, result_bytes, cudaMemcpyDeviceToHost, stream));
            cuda_check(cudaStreamSynchronize(stream));
            for (const auto& m : models) {
                const auto count    = m.network->classes.size();
                const auto base     = host_results + m.offset;
                const auto scores   = reinterpret_cast<const float*>(base);
                const int index     = *reinterpret_cast<const int*>(base + count * 4);
                const bool accepted = base[count * 4 + 4] != 0;
                result.classifiers.push_back({m.descriptor.id, m.network->classes[index], m.network->classes, {scores, scores + count}, m.threshold, accepted, {}});
                result.passed &= accepted;
            }
        } catch (const std::exception& failure) {
            result.passed = false;
            result.error  = failure.what();
            for (const auto& m : models) result.classifiers.push_back({m.descriptor.id, "", m.network->classes, {}, m.threshold, false, result.error});
        }
        return result;
    }
} // namespace classifier
