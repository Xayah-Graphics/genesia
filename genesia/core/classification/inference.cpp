module;
#include "../models/convnext/kernels.h"
#include <nlohmann/json.hpp>
module genesia.classification.inference;
import genesia.models.registry;
import genesia.project;
import genesia.io.files;
import genesia.data.images;
import genesia.training.samples;
import std;
namespace genesia::classification {
    namespace {
        // Windows exposes the new name before MoveFileEx closes its rename handle.
        std::mutex cache_files;
    } // namespace

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
    Pipeline::~Pipeline() {
        if (pending_stream) cudaStreamSynchronize(pending_stream);
        for (auto& model : models) model.plans.clear();
        arena.reset();
        models.clear();
        if (host_results) cudaFreeHost(host_results);
    }
    bool Pipeline::prepare(std::vector<models::Descriptor> desired, int width, int height) {
        std::ranges::sort(desired, {}, &models::Descriptor::id);
        if (!desired.empty() && !((width == 1024 && height == 1536) || (width == 1536 && height == 1024) || (width == 1024 && height == 1024))) throw std::runtime_error("Enabled classifiers require 1024x1536, 1536x1024, or 1024x1024 original RGB8 input");
        const bool changed = (!models.empty() && !arena) || desired.size() != models.size() || !std::equal(desired.begin(), desired.end(), models.begin(), [](const models::Descriptor& a, const LoadedClassifier& b) { return a == b.descriptor; });
        if (changed) {
            if (pending_stream) compute::check(cudaStreamSynchronize(pending_stream));
            pending_stream = nullptr;
            for (auto& model : models) model.plans.clear();
            arena.reset();
            std::erase_if(models, [&](const LoadedClassifier& model) { return !std::ranges::contains(desired, model.descriptor); });
            for (const auto& descriptor : desired) {
                if (std::ranges::contains(models, descriptor, &LoadedClassifier::descriptor)) continue;
                LoadedClassifier entry;
                entry.descriptor = descriptor;
                entry.network    = std::make_unique<convnext::Network>(descriptor.path);
                models.push_back(std::move(entry));
            }
            std::ranges::sort(models, {}, [](const LoadedClassifier& model) { return model.descriptor.id; });
            result_bytes = 0;
            for (auto& m : models) {
                m.offset = result_bytes;
                result_bytes += (m.network->classes.size() * 4 + 4 + 255) / 256 * 256;
            }
            result_storage = compute::DeviceBuffer(result_bytes);
            if (host_results) cudaFreeHost(std::exchange(host_results, nullptr));
            if (result_bytes) compute::check(cudaMallocHost(&host_results, result_bytes));
            for (auto& m : models) {
                auto base        = static_cast<std::byte*>(result_storage.data) + m.offset;
                const auto count = m.network->classes.size();
                m.output         = {reinterpret_cast<float*>(base), reinterpret_cast<int*>(base + count * 4)};
            }
            if (!models.empty()) {
                const auto largest = std::ranges::max_element(models, {}, [](const LoadedClassifier& m) { return m.network->classes.size(); });
                arena              = std::make_unique<convnext::NetworkPlan>(*largest->network, 1, 1024, 1536);
            }
        }
        for (auto& m : models) {
            const auto shape = std::pair{width, height};
            if (!m.plans.contains(shape)) {
                auto plan = std::make_unique<convnext::NetworkPlan>(*m.network, 1, width, height, convnext::ActivationStorage::reuse, arena.get());
                plan->capture();
                m.plans.emplace(shape, std::move(plan));
            }
        }
        return changed;
    }
    Result Pipeline::run(const models::Descriptor& descriptor, convnext::GpuRgb8 input, cudaStream_t stream, const std::string_view image_sha) {
        pending_stream = stream;
        auto& m        = *std::ranges::find(models, descriptor, &LoadedClassifier::descriptor);
        m.plans.at({input.width, input.height})->infer(input, m.output, stream);
        const auto count = m.network->classes.size();
        compute::check(cudaMemcpyAsync(host_results + m.offset, static_cast<std::byte*>(result_storage.data) + m.offset, count * 4 + 4, cudaMemcpyDeviceToHost, stream));
        compute::check(cudaStreamSynchronize(stream));
        const auto base   = host_results + m.offset;
        const auto scores = reinterpret_cast<const float*>(base);
        const int index   = *reinterpret_cast<const int*>(base + count * 4);
        return {m.descriptor.id, m.descriptor.sha, std::string(image_sha), m.network->classes[index], m.network->classes, {scores, scores + count}};
    }
    std::optional<Result> Cache::find(const models::Descriptor& descriptor, const std::string_view image_sha) {
        const auto key = std::pair{descriptor.sha, std::string{image_sha}};
        if (const auto found = entries.find(key); found != entries.end()) {
            auto result = found->second;
            result.id   = descriptor.id;
            return result;
        }
        const std::lock_guard lock{cache_files};
        const auto path = std::filesystem::path{project::cache} / "inference" / descriptor.sha / (std::string{image_sha} + ".json");
        if (!std::filesystem::exists(path)) return {};
        const auto value = files::read_json(path);
        if (value.at("version") != 1) throw std::runtime_error{"Unsupported prediction cache format"};
        auto result  = value.at("prediction").get<Result>();
        result.id    = descriptor.id;
        entries[key] = result;
        return result;
    }
    void Cache::store(const Result& result) {
        const std::lock_guard lock{cache_files};
        const auto path = std::filesystem::path{project::cache} / "inference" / result.model_sha / (result.image_sha + ".json");
        files::write_json(path, {{"version", 1}, {"prediction", result}});
        entries[{result.model_sha, result.image_sha}] = result;
    }
    Result Predictions::infer(const models::Descriptor& descriptor, const dataset::File& file, const bool refresh, const std::span<const std::uint8_t> rgb) {
        if (!refresh)
            if (auto result = cache.find(descriptor, file.sha)) return *result;
        std::vector<models::Descriptor> desired;
        for (const auto& model : pipeline.models)
            if (model.descriptor.id != descriptor.id && std::ranges::contains(active, model.descriptor.id)) desired.push_back(model.descriptor);
        desired.push_back(descriptor);
        if (pipeline.prepare(std::move(desired), file.width, file.height)) pixel_sha.clear();
        auto& loaded = *std::ranges::find(pipeline.models, descriptor, &LoadedClassifier::descriptor);
        auto& plan   = *loaded.plans.at({file.width, file.height});
        if (pixel_sha != file.sha) {
            if (rgb.empty() && image_sha != file.sha) {
                image     = read_image(file.path);
                image_sha = file.sha;
            }
            const auto bytes = rgb.empty() ? std::span<const std::uint8_t>{image.pixels} : rgb;
            compute::check(cudaMemcpyAsync(plan.pixels.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice, plan.stream));
            compute::check(cudaStreamSynchronize(plan.stream));
            pixel_sha = file.sha;
        }
        auto result = pipeline.run(descriptor, {static_cast<unsigned char*>(plan.pixels.data), file.width, file.height, std::size_t(file.width) * 3}, plan.stream, file.sha);
        cache.store(result);
        return result;
    }
} // namespace genesia::classification
