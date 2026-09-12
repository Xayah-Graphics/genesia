module;
#include "kernels.h"

#include <nlohmann/json.hpp>
module classifier.training;
import std;
namespace classifier {
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainingConfig, physical_batch, effective_batch, head_only_steps, warmup_steps, head_only_lr, backbone_lr, head_lr, weight_decay, clip_norm, seed, eval_interval, save_interval, log_interval)
    struct TrainingResources {
        cudaStream_t stream       = nullptr;
        cudaGraph_t graph         = nullptr;
        cudaGraphExec_t optimizer = nullptr;
        std::array<unsigned char*, 2> pinned{};
        std::array<cudaEvent_t, 2> copied{};
        TrainingResources(std::size_t capacity, int physical) {
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            for (int i = 0; i < 2; ++i) {
                cuda_check(cudaMallocHost(&pinned[i], capacity + physical * 4));
                cuda_check(cudaEventCreateWithFlags(&copied[i], cudaEventDisableTiming));
            }
        }
        ~TrainingResources() {
            cudaStreamSynchronize(stream);
            for (int i = 0; i < 2; ++i) {
                cudaFreeHost(pinned[i]);
                cudaEventDestroy(copied[i]);
            }
            if (optimizer) cudaGraphExecDestroy(optimizer);
            if (graph) cudaGraphDestroy(graph);
            cudaStreamDestroy(stream);
        }
    };
    nlohmann::json evaluate(Network& model, Dataset& dataset, const std::filesystem::path& output, int step) {
        std::map<std::pair<int, int>, std::unique_ptr<NetworkPlan>> plans;
        int count = int(model.classes.size());
        DeviceBuffer result(count * 4 + 8);
        GpuResult gpu{static_cast<float*>(result.data), reinterpret_cast<int*>(static_cast<std::byte*>(result.data) + count * 4), reinterpret_cast<unsigned char*>(static_cast<std::byte*>(result.data) + count * 4 + 4)};
        nlohmann::json predictions = nlohmann::json::array();
        std::vector<std::vector<int>> confusion(count, std::vector<int>(count));
        double loss = 0, bce = 0;
        std::array<std::array<int, 4>, 3> binary{};
        constexpr float thresholds[3] = {.5f, .7f, .9f};
        for (int index : dataset.val) {
            const auto& r = dataset.records[index];
            auto shape    = std::pair{r.width, r.height};
            if (!plans.contains(shape)) {
                plans[shape] = std::make_unique<NetworkPlan>(model, 1, r.width, r.height);
                plans[shape]->capture();
            }
            auto& plan = *plans[shape];
            cuda_check(cudaMemcpyAsync(plan.pixels.data, dataset.rgb(index), plan.pixels.size, cudaMemcpyHostToDevice, plan.stream));
            plan.infer({static_cast<unsigned char*>(plan.pixels.data), r.width, r.height, std::size_t(r.width) * 3}, gpu, plan.stream);
            std::vector<float> scores(count);
            cuda_check(cudaMemcpyAsync(scores.data(), gpu.scores, count * 4, cudaMemcpyDeviceToHost, plan.stream));
            cuda_check(cudaStreamSynchronize(plan.stream));
            int predicted = int(std::max_element(scores.begin(), scores.end()) - scores.begin());
            ++confusion[r.label][predicted];
            loss -= std::log(std::max(scores[r.label], 1e-30f));
            bce -= std::log(std::max(r.label == 0 ? scores[0] : 1 - scores[0], 1e-30f));
            for (int t = 0; t < 3; ++t) ++binary[t][r.label == 0 ? (scores[0] >= thresholds[t] ? 0 : 2) : (scores[0] >= thresholds[t] ? 1 : 3)];
            predictions.push_back({{"id", r.id}, {"label", r.label}, {"scores", std::vector<double>(scores.begin(), scores.end())}});
        }
        nlohmann::json metrics = {{"step", step}, {"loss", loss / dataset.val.size()}, {"yes_bce", bce / dataset.val.size()}, {"confusion", confusion}, {"samples", dataset.val.size()}};
        for (int t = 0; t < 3; ++t) {
            auto [tp, fp, fn, tn] = binary[t];
            double precision = tp ? double(tp) / (tp + fp) : 0, recall = tp ? double(tp) / (tp + fn) : 0;
            metrics[std::format("{:.1f}", thresholds[t])] = {{"tp", tp}, {"fp", fp}, {"fn", fn}, {"tn", tn}, {"precision", precision}, {"recall", recall}, {"f0.5", precision + recall ? 1.25 * precision * recall / (.25 * precision + recall) : 0}};
        }
        write_json(output / std::format("step_{:04}_predictions.json", step), predictions);
        write_json(output / std::format("step_{:04}_metrics.json", step), metrics);
        std::println("Evaluation {}: CE {:.5f}, threshold .9 {}", step, loss / dataset.val.size(), metrics["0.9"].dump());
        return metrics;
    }
    void train(const TrainingOptions& options, const std::atomic_bool& interrupted) {
        auto root = std::filesystem::absolute(options.dataset), latest = root / "latest.safetensors";
        std::string content = fingerprint(root);
        nlohmann::json state, saved_config = TrainingConfig{};
        int completed = 0;
        if (std::filesystem::exists(latest)) {
            SafeFile saved(latest);
            auto meta    = saved.header.at("__metadata__");
            state        = nlohmann::json::parse(meta.at("training_state").get<std::string>());
            completed    = std::stoi(meta.at("step").get<std::string>());
            saved_config = state.at("config");
        }
        bool resume           = !state.is_null() && !options.restart && state.at("fingerprint") == content;
        nlohmann::json config = saved_config;
        for (auto item = options.overrides.begin(); item != options.overrides.end(); ++item) {
            bool frequency = item.key() == "eval_interval" || item.key() == "save_interval" || item.key() == "log_interval";
            if (resume && !frequency && item.value() != saved_config.at(item.key())) throw std::runtime_error("Changing " + item.key() + " requires --restart");
            config[item.key()] = item.value();
        }
        TrainingConfig settings = config.get<TrainingConfig>();
        if (resume && completed >= options.steps) {
            std::println("Already at step {}; target {} reached.", completed, options.steps);
            return;
        }
        if (!resume) {
            if (!state.is_null()) {
                std::string old = state.at("run_id");
                for (const auto* name : std::array{"latest", "best", "model"}) {
                    auto file = root / (std::string(name) + ".safetensors");
                    if (std::filesystem::exists(file)) std::filesystem::rename(file, root / std::format("{}.{}.safetensors", name, old));
                }
            }
            std::string run = std::format("{}-{}", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(), content.substr(0, 8));
            state           = {{"run_id", run}, {"fingerprint", content}, {"schedule_steps", options.steps}, {"cursor", 0}, {"best", {-1., -1., -1.}}};
            completed       = 0;
        }
        state["config"] = config;
        auto snapshot   = root / ".classifier" / "data" / content;
        if (!std::filesystem::exists(snapshot / "dataset.json")) {
            auto temporary = snapshot;
            temporary += std::format(".building-{}", state.at("run_id").get<std::string>());
            prepare(categories(root), temporary);
            std::filesystem::rename(temporary, snapshot);
        }
        Dataset dataset(snapshot);
        state["split"] = dataset.metadata;
        if (dataset.train.empty() || dataset.val.empty()) throw std::runtime_error("Grouped split needs independent images in both train and validation sets");
        auto output = root / ".classifier" / state.at("run_id").get<std::string>();
        std::filesystem::create_directories(output);
        Network model(resume ? latest : std::filesystem::path(u8"" CLASSIFIER_INITIAL_MODEL), resume ? NetworkLoad::resume : NetworkLoad::pretrained);
        model.cache_directory = root / ".classifier" / "cache";
        if (!resume) model.initialize_head(dataset.classes, settings.seed);
        model.metadata["threshold"] = "0.9";
        UpdateState update;
        update.step            = completed;
        update.schedule_steps  = state.at("schedule_steps");
        update.head_only_steps = settings.head_only_steps;
        update.effective_batch = settings.effective_batch;
        update.warmup_steps    = settings.warmup_steps;
        update.head_only_lr    = settings.head_only_lr;
        update.backbone_lr     = settings.backbone_lr;
        update.head_lr         = settings.head_lr;
        update.weight_decay    = settings.weight_decay;
        update.clip_norm       = settings.clip_norm;
        RandomState rng{settings.seed, resume ? std::stoull(model.metadata.at("sequence").get<std::string>()) : 0};
        cuda_check(cudaMemcpy(model.optimizer->random, &rng, sizeof(rng), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(model.optimizer->update, &update, sizeof(update), cudaMemcpyHostToDevice));
        int physical         = settings.physical_batch;
        std::size_t capacity = std::size_t(physical) * 1536 * 1024 * 3;
        TrainingResources resources(capacity, physical);
        std::map<std::tuple<int, int, int>, std::unique_ptr<TrainingPlan>> plans;
        auto largest = std::make_unique<TrainingPlan>(model, physical, 1024, 1536);
        largest->capture_training();
        TrainingPlan* shared          = largest.get();
        plans[{physical, 1024, 1536}] = std::move(largest);
        auto get_plan                 = [&](int n, int w, int h) -> TrainingPlan& {
            auto key = std::tuple{n, w, h};
            if (!plans.contains(key)) {
                plans[key] = std::make_unique<TrainingPlan>(model, n, w, h, shared);
                plans[key]->capture_training();
            }
            return *plans.at(key);
        };
        std::vector<int> counts(model.classes.size());
        for (int i : dataset.train) ++counts[dataset.records[i].label];
        std::vector<double> weights;
        for (int i : dataset.train) weights.push_back(1. / std::sqrt(counts[dataset.records[i].label]));
        std::mt19937_64 random(settings.seed);
        std::discrete_distribution<int> sampler(weights.begin(), weights.end());
        std::uint64_t cursor = state.at("cursor").get<std::uint64_t>();
        for (std::uint64_t i = 0; i < cursor; ++i) static_cast<void>(sampler(random));
        cuda_check(cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeThreadLocal));
        for (auto& [name, p] : model.parameters) gradient_norm(resources.stream, p.gpu, model.optimizer->update);
        optimizer_clip(resources.stream, model.optimizer->update);
        for (auto& [name, p] : model.parameters) adamw(resources.stream, p.gpu, model.optimizer->update);
        cuda_check(cudaStreamEndCapture(resources.stream, &resources.graph));
        cuda_check(cudaGraphInstantiate(&resources.optimizer, resources.graph, 0));
        nlohmann::json history = nlohmann::json::array();
        if (resume && std::filesystem::exists(output / "history.json")) {
            std::ifstream saved(output / "history.json");
            saved >> history;
        }
        auto checkpoint = [&](bool evaluation) {
            state["cursor"] = cursor;
            if (evaluation) {
                auto metrics = evaluate(model, dataset, output, update.step);
                history.push_back(metrics);
                std::array<double, 3> score{metrics["0.9"]["f0.5"], metrics["0.9"]["precision"], -metrics["yes_bce"].get<double>()};
                if (score > state.at("best").get<std::array<double, 3>>()) {
                    state["best"] = score;
                    model.save(root / "best.safetensors", state);
                    model.save(root / "model.safetensors", state, false);
                }
                write_json(output / "history.json", history);
            }
            model.save(latest, state);
        };
        if (!resume) checkpoint(true);
        std::ofstream log(output / "steps.csv", resume ? std::ios::app : std::ios::out);
        if (!resume) log << "step,loss,gradient_norm,seconds\n";
        auto start = std::chrono::steady_clock::now();
        while (update.step < options.steps && !interrupted.load(std::memory_order_relaxed)) {
            std::map<std::pair<int, int>, std::vector<int>> batch;
            for (int i = 0; i < settings.effective_batch; ++i, ++cursor) {
                int index     = dataset.train[sampler(random)];
                const auto& r = dataset.records[index];
                batch[{r.width, r.height}].push_back(index);
            }
            for (const auto& [shape, indices] : batch)
                for (int begin = 0; begin < int(indices.size()); begin += physical) get_plan(std::min(physical, int(indices.size()) - begin), shape.first, shape.second);
            optimizer_begin(resources.stream, model.optimizer->update);
            ++update.step;
            int micro = 0;
            for (const auto& [shape, indices] : batch)
                for (int begin = 0; begin < int(indices.size()); begin += physical) {
                    int n             = std::min(physical, int(indices.size()) - begin);
                    auto& plan        = get_plan(n, shape.first, shape.second);
                    std::size_t bytes = std::size_t(shape.first) * shape.second * 3;
                    int slot          = micro++ % 2;
                    cuda_check(cudaEventSynchronize(resources.copied[slot]));
                    auto targets = reinterpret_cast<int*>(resources.pinned[slot] + capacity);
                    for (int j = 0; j < n; ++j) {
                        int index = indices[begin + j];
                        std::memcpy(resources.pinned[slot] + j * bytes, dataset.rgb(index), bytes);
                        targets[j] = dataset.records[index].label;
                    }
                    plan.microbatch(resources.pinned[slot], targets, update.step <= settings.head_only_steps, resources.stream, resources.copied[slot]);
                }
            cuda_check(cudaGraphLaunch(resources.optimizer, resources.stream));
            cuda_check(cudaMemcpyAsync(&update, model.optimizer->update, sizeof(update), cudaMemcpyDeviceToHost, resources.stream));
            cuda_check(cudaStreamSynchronize(resources.stream));
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            log << update.step << ',' << update.loss << ',' << std::sqrt(update.norm2) << ',' << seconds << '\n';
            log.flush();
            if (update.step % settings.log_interval == 0) {
                std::println("Step {} loss {:.6f} norm {:.5f} elapsed {:.1f}s", update.step, update.loss, std::sqrt(update.norm2), seconds);
                std::fflush(stdout);
            }
            bool final = update.step == options.steps || interrupted.load(std::memory_order_relaxed), evaluation = final || update.step % settings.eval_interval == 0;
            if (evaluation || update.step % settings.save_interval == 0) checkpoint(evaluation);
        }
        const bool stopped = interrupted.load(std::memory_order_relaxed);
        if (stopped) checkpoint(false);
        std::println("{} at step {}. Checkpoints: {}", stopped ? "Interrupted" : "Completed", update.step, path_utf8(root));
    }
} // namespace classifier
