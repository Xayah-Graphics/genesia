module;
#include "models/convnext/kernels.h"
#include <nlohmann/json.hpp>
module classifier.inference;
import classifier.project;
import std;
namespace classifier::classification {
    void to_json(nlohmann::json& json, const Result& value) {
        json = {{"model_sha", value.model_sha}, {"image_sha", value.image_sha}, {"label", value.label}, {"classes", value.classes}, {"scores", value.scores}};
    }
    void from_json(const nlohmann::json& json, Result& value) {
        json.at("model_sha").get_to(value.model_sha);
        json.at("image_sha").get_to(value.image_sha);
        json.at("label").get_to(value.label);
        json.at("classes").get_to(value.classes);
        json.at("scores").get_to(value.scores);
    }
    std::optional<Result> Cache::find(const models::Descriptor& model, const std::string_view image_sha) {
        const auto key = std::pair{model.sha, std::string{image_sha}};
        if (const auto found = entries.find(key); found != entries.end()) return found->second;
        const auto path = project::cache / "inference" / model.sha / (std::string{image_sha} + ".json");
        if (!std::filesystem::exists(path)) return {};
        auto result  = tools::files::read_json(path).get<Result>();
        entries[key] = result;
        return result;
    }
    void Cache::store(const Result& result) {
        tools::files::write_json(project::cache / "inference" / result.model_sha / (result.image_sha + ".json"), result);
        entries[{result.model_sha, result.image_sha}] = result;
    }
    Result Predictions::infer(const models::Descriptor& model, const dataset::File& file, const bool refresh, const std::function<void(const runtime::Progress&)>& progress) {
        if (!refresh)
            if (auto cached = cache.find(model, file.sha)) return *cached;
        if (descriptor != model) {
            progress({runtime::BatchProgress{runtime::Stage::loading_model}});
            plans.clear();
            arena.reset();
            network    = std::make_unique<convnext::Network>(model.path);
            arena      = std::make_unique<convnext::NetworkPlan>(*network, 1, 1024, 1536);
            output     = tools::compute::DeviceBuffer{network->classes.size() * sizeof(float) + sizeof(int)};
            descriptor = model;
        }
        const auto shape = std::pair{file.width, file.height};
        if (!plans.contains(shape)) {
            progress({runtime::BatchProgress{runtime::Stage::preparing_inference}});
            auto plan = std::make_unique<convnext::NetworkPlan>(*network, 1, file.width, file.height, convnext::ActivationStorage::reuse, arena.get());
            plan->capture();
            plans.emplace(shape, std::move(plan));
        }
        auto& plan       = *plans.at(shape);
        const auto image = tools::read_image(file.path);
        tools::compute::check(cudaMemcpyAsync(plan.pixels.data, image.pixels.data(), image.pixels.size(), cudaMemcpyHostToDevice, plan.stream));
        const auto count = network->classes.size();
        const convnext::GpuResult gpu{static_cast<float*>(output.data), reinterpret_cast<int*>(static_cast<std::byte*>(output.data) + count * sizeof(float))};
        plan.infer({static_cast<unsigned char*>(plan.pixels.data), file.width, file.height, std::size_t(file.width) * 3}, gpu, plan.stream);
        Result result{model.sha, file.sha, {}, network->classes, std::vector<float>(count)};
        int predicted{};
        tools::compute::check(cudaMemcpyAsync(result.scores.data(), gpu.scores, count * sizeof(float), cudaMemcpyDeviceToHost, plan.stream));
        tools::compute::check(cudaMemcpyAsync(&predicted, gpu.indices, sizeof(predicted), cudaMemcpyDeviceToHost, plan.stream));
        tools::compute::check(cudaStreamSynchronize(plan.stream));
        result.label = result.classes[predicted];
        cache.store(result);
        return result;
    }
} // namespace classifier::classification
