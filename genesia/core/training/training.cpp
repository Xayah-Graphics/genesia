module;
#include "../models/convnext/kernels.h"
#include <nlohmann/json.hpp>
module genesia.training;
import genesia.models.convnext;
import std;
import genesia.io.files;
import genesia.io.hash;
import genesia.data.transactions;
namespace genesia::training {
    struct Resume final {
        std::string fingerprint;
        int schedule_steps{};
        Config config;
        Snapshot split;
        std::string sampler;
    };
    void to_json(nlohmann::json& json, const Resume& value) {
        json = {{"fingerprint", value.fingerprint}, {"schedule_steps", value.schedule_steps}, {"config", value.config}, {"split", value.split}, {"sampler", value.sampler}};
    }
    void from_json(const nlohmann::json& json, Resume& value) {
        json.at("fingerprint").get_to(value.fingerprint);
        json.at("schedule_steps").get_to(value.schedule_steps);
        json.at("config").get_to(value.config);
        json.at("split").get_to(value.split);
        json.at("sampler").get_to(value.sampler);
    }
    struct TrainingResources {
        cudaStream_t stream       = nullptr;
        cudaGraph_t graph         = nullptr;
        cudaGraphExec_t optimizer = nullptr;
        std::array<unsigned char*, 2> pinned{};
        std::array<cudaEvent_t, 2> copied{};
        TrainingResources(std::size_t capacity, int physical) {
            compute::check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            for (int i = 0; i < 2; ++i) {
                compute::check(cudaMallocHost(&pinned[i], capacity + physical * 4));
                compute::check(cudaEventCreateWithFlags(&copied[i], cudaEventDisableTiming));
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
    Metrics evaluate(convnext::Network& model, Dataset& dataset, int step) {
        std::map<std::pair<int, int>, std::unique_ptr<convnext::NetworkPlan>> plans;
        int count = int(model.classes.size());
        compute::DeviceBuffer result(count * 4 + 4);
        convnext::GpuResult gpu{static_cast<float*>(result.data), reinterpret_cast<int*>(static_cast<std::byte*>(result.data) + count * 4)};
        std::vector<std::vector<int>> confusion(count, std::vector<int>(count));
        double loss = 0;
        for (int index : dataset.val) {
            const auto& r = dataset.snapshot.records[index];
            auto shape    = std::pair{r.width, r.height};
            if (!plans.contains(shape)) {
                plans[shape] = std::make_unique<convnext::NetworkPlan>(model, 1, r.width, r.height);
                plans[shape]->capture();
            }
            auto& plan = *plans[shape];
            compute::check(cudaMemcpyAsync(plan.pixels.data, dataset.rgb(index), plan.pixels.size, cudaMemcpyHostToDevice, plan.stream));
            plan.infer({static_cast<unsigned char*>(plan.pixels.data), r.width, r.height, std::size_t(r.width) * 3}, gpu, plan.stream);
            std::vector<float> scores(count);
            compute::check(cudaMemcpyAsync(scores.data(), gpu.scores, count * 4, cudaMemcpyDeviceToHost, plan.stream));
            compute::check(cudaStreamSynchronize(plan.stream));
            int predicted = int(std::max_element(scores.begin(), scores.end()) - scores.begin());
            ++confusion[r.label][predicted];
            loss -= std::log(std::max(scores[r.label], 1e-30f));
        }
        double precision = 0, recall = 0, f1 = 0;
        int correct = 0;
        std::vector<ClassMetric> per_class;
        for (int label = 0; label < count; ++label) {
            int actual = std::accumulate(confusion[label].begin(), confusion[label].end(), 0), predicted = 0;
            for (const auto& row : confusion) predicted += row[label];
            const int tp   = confusion[label][label];
            const double p = predicted ? double(tp) / predicted : 0;
            const double r = actual ? double(tp) / actual : 0;
            const double f = p + r ? 2 * p * r / (p + r) : 0;
            precision += p;
            recall += r;
            f1 += f;
            correct += tp;
            per_class.push_back({model.classes[label], p, r, f, static_cast<std::size_t>(actual)});
        }
        return {step, loss / dataset.val.size(), double(correct) / dataset.val.size(), precision / count, recall / count, f1 / count, std::move(per_class), std::move(confusion), dataset.val.size()};
    }
    State train(const Options& options, TrainingData source, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress) {
        const auto key_bytes = std::span{reinterpret_cast<const unsigned char*>(options.concept_key.data()), options.concept_key.size()};
        auto assigned        = dataset::read_concept(options.concept_key);
        if (assigned.type == dataset::ConceptType::none) throw std::runtime_error{"Assign a concept type before training"};
        if (assigned.type == dataset::ConceptType::lora) throw std::runtime_error{"LoRA training is not implemented"};
        if (!source.issue.empty()) throw std::runtime_error{source.issue};
        if (!source.training_issue.empty()) throw std::runtime_error{source.training_issue};
        const auto root       = source.root / ".genesia" / "training";
        const auto latest     = root / "checkpoint.safetensors";
        const auto model_path = root / "publishing.safetensors";
        const auto snapshot   = std::filesystem::path{GENESIA_CACHE_DIRECTORY} / "rgb" / sha256(key_bytes) / source.fingerprint;
        std::optional<Resume> saved_state;
        Config saved_config;
        int completed = 0;
        if (!options.restart && source.training.has_value()) {
            saved_config = source.training->config;
            if (source.training->fingerprint != source.fingerprint) throw std::runtime_error{"Training data changed; Restart training is required"};
            if (std::filesystem::exists(latest)) {
                files::SafeFile saved(latest);
                const auto& meta = saved.header.at("__metadata__");
                if (meta.at("format") != "genesia-convnext-2") throw std::runtime_error{"Unsupported classifier checkpoint"};
                saved_state  = nlohmann::json::parse(meta.at("training_state").get<std::string>()).get<Resume>();
                completed    = std::stoi(meta.at("step").get<std::string>());
                saved_config = saved_state->config;
            } else throw std::runtime_error{"Training checkpoint is missing; Restart training is required"};
        }
        const bool resume     = saved_state.has_value();
        const Config settings = options.config.value_or(options.restart ? Config{} : saved_config);
        if (!options.restart && source.training && settings != saved_config) throw std::runtime_error{"Training parameters changed; Restart training is required"};
        if (options.steps <= completed || options.steps <= 0) throw std::runtime_error{"Target steps must exceed the completed step"};
        if (!options.restart && source.training.has_value() && options.steps < source.training->target) throw std::runtime_error{"Cumulative target steps cannot decrease"};
        if (settings.physical_batch <= 0 || settings.physical_batch > 256 || settings.effective_batch <= 0 || settings.eval_interval <= 0 || settings.save_interval <= 0 || settings.log_interval <= 0) throw std::runtime_error{"Invalid training batch or interval"};
        if (!std::filesystem::exists(snapshot / "snapshot.json")) {
            const auto temporary = std::filesystem::path{snapshot.string() + ".building"};
            if (std::filesystem::exists(temporary)) std::filesystem::remove_all(temporary);
            try {
                prepare(source, temporary, interrupted, [&](std::size_t done, std::size_t total) { progress({runtime::BatchProgress{runtime::Stage::preparing, done, total}}); });
                std::filesystem::rename(temporary, snapshot);
            } catch (...) {
                std::filesystem::remove_all(temporary);
                throw;
            }
        }
        if (interrupted.load()) throw runtime::Stopped{};
        if (options.restart && std::filesystem::exists(root)) {
            if (!source.training) throw std::runtime_error{"Cannot reset an unregistered concept state directory"};
            source.model.reset();
            models::unpublish(options.concept_key);
            std::filesystem::remove_all(root);
            std::filesystem::remove(source.root / ".genesia" / "audit-moves.json");
            for (const auto& entry : std::filesystem::directory_iterator(snapshot.parent_path()))
                if (entry.path() != snapshot) std::filesystem::remove_all(entry.path());
            source.training.reset();
        }
        std::filesystem::create_directories(root);
        State status{source.fingerprint, source.classes, Phase::preparing, options.steps, completed, settings};
        if (!assigned.locked) {
            assigned.locked = true;
            files::write_json(assigned.path / ".genesia" / "concept.json", assigned);
        }
        write_state(source.root, status);
        try {
            Dataset dataset(snapshot);
            files::write_json(root / "snapshot.json", dataset.snapshot);
            auto state = saved_state.value_or(Resume{source.fingerprint, options.steps, settings, dataset.snapshot});
            if (resume && state.split != dataset.snapshot) throw std::runtime_error{"Training split changed; Restart training is required"};
            convnext::Network model(resume ? latest : std::filesystem::path{u8"" CLASSIFIER_INITIAL_MODEL}, resume ? convnext::NetworkLoad::resume : convnext::NetworkLoad::pretrained);
            if (!resume) model.initialize_head(dataset.snapshot.classes, settings.seed);
            convnext::UpdateState update;
            update.step            = completed;
            update.schedule_steps  = state.schedule_steps;
            update.head_only_steps = settings.head_only_steps;
            update.effective_batch = settings.effective_batch;
            update.warmup_steps    = settings.warmup_steps;
            update.head_only_lr    = settings.head_only_lr;
            update.backbone_lr     = settings.backbone_lr;
            update.head_lr         = settings.head_lr;
            update.weight_decay    = settings.weight_decay;
            update.clip_norm       = settings.clip_norm;
            convnext::RandomState rng{settings.seed, resume ? model.sequence : 0};
            compute::check(cudaMemcpy(model.optimizer->random, &rng, sizeof(rng), cudaMemcpyHostToDevice));
            compute::check(cudaMemcpy(model.optimizer->update, &update, sizeof(update), cudaMemcpyHostToDevice));
            int physical         = settings.physical_batch;
            std::size_t capacity = std::size_t(physical) * 1536 * 1024 * 3;
            TrainingResources resources(capacity, physical);
            std::map<std::tuple<int, int, int>, std::unique_ptr<convnext::TrainingPlan>> plans;
            auto largest = std::make_unique<convnext::TrainingPlan>(model, physical, 1024, 1536);
            largest->capture_training();
            convnext::TrainingPlan* shared = largest.get();
            plans[{physical, 1024, 1536}]  = std::move(largest);
            auto get_plan                  = [&](int n, int w, int h) -> convnext::TrainingPlan& {
                auto key = std::tuple{n, w, h};
                if (!plans.contains(key)) {
                    plans[key] = std::make_unique<convnext::TrainingPlan>(model, n, w, h, shared);
                    plans[key]->capture_training();
                }
                return *plans.at(key);
            };
            std::vector<int> counts(model.classes.size());
            for (int i : dataset.train) ++counts[dataset.snapshot.records[i].label];
            std::vector<double> weights;
            for (int i : dataset.train) weights.push_back(1. / std::sqrt(counts[dataset.snapshot.records[i].label]));
            std::mt19937_64 random(settings.seed);
            std::discrete_distribution<int> sampler(weights.begin(), weights.end());
            if (resume) {
                std::istringstream saved{state.sampler};
                saved >> random;
            }
            compute::check(cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeThreadLocal));
            for (auto& [name, p] : model.parameters) convnext::gradient_norm(resources.stream, p.gpu, model.optimizer->update);
            convnext::optimizer_clip(resources.stream, model.optimizer->update);
            for (auto& [name, p] : model.parameters) convnext::adamw(resources.stream, p.gpu, model.optimizer->update);
            compute::check(cudaStreamEndCapture(resources.stream, &resources.graph));
            compute::check(cudaGraphInstantiate(&resources.optimizer, resources.graph, 0));
            auto checkpoint = [&](bool evaluation) {
                std::ostringstream saved_random;
                saved_random << random;
                state.sampler = saved_random.str();
                if (evaluation) {
                    auto metrics = evaluate(model, dataset, update.step);
                    record_metric(source.root, metrics);
                    progress({metrics});
                }
                model.save(latest, nlohmann::json(state).dump());
                status.step       = update.step;
                status.phase      = Phase::training;
                status.checkpoint = true;
                write_state(source.root, status);
            };
            if (!resume) checkpoint(true);
            auto start = std::chrono::steady_clock::now();
            while (update.step < options.steps && !interrupted.load(std::memory_order_relaxed)) {
                std::map<std::pair<int, int>, std::vector<int>> batch;
                for (int i = 0; i < settings.effective_batch; ++i) {
                    int index     = dataset.train[sampler(random)];
                    const auto& r = dataset.snapshot.records[index];
                    batch[{r.width, r.height}].push_back(index);
                }
                for (const auto& [shape, indices] : batch)
                    for (int begin = 0; begin < int(indices.size()); begin += physical) get_plan(std::min(physical, int(indices.size()) - begin), shape.first, shape.second);
                convnext::optimizer_begin(resources.stream, model.optimizer->update);
                ++update.step;
                int micro = 0;
                for (const auto& [shape, indices] : batch) {
                    for (int begin = 0; begin < int(indices.size()); begin += physical) {
                        int n             = std::min(physical, int(indices.size()) - begin);
                        auto& plan        = get_plan(n, shape.first, shape.second);
                        std::size_t bytes = std::size_t(shape.first) * shape.second * 3;
                        int slot          = micro++ % 2;
                        compute::check(cudaEventSynchronize(resources.copied[slot]));
                        auto targets = reinterpret_cast<int*>(resources.pinned[slot] + capacity);
                        for (int j = 0; j < n; ++j) {
                            int index = indices[begin + j];
                            std::memcpy(resources.pinned[slot] + j * bytes, dataset.rgb(index), bytes);
                            targets[j] = dataset.snapshot.records[index].label;
                        }
                        plan.microbatch(resources.pinned[slot], targets, update.step <= settings.head_only_steps, resources.stream, resources.copied[slot]);
                    }
                }
                compute::check(cudaGraphLaunch(resources.optimizer, resources.stream));
                compute::check(cudaMemcpyAsync(&update, model.optimizer->update, sizeof(update), cudaMemcpyDeviceToHost, resources.stream));
                compute::check(cudaStreamSynchronize(resources.stream));
                double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                const Step metric{update.step, options.steps, update.loss, std::sqrt(update.norm2), seconds};
                record_metric(source.root, metric);
                if (update.step % settings.log_interval == 0) progress({metric});
                bool final      = update.step == options.steps || interrupted.load(std::memory_order_relaxed);
                bool evaluation = final || update.step % settings.eval_interval == 0;
                if (evaluation || update.step % settings.save_interval == 0) checkpoint(evaluation);
            }
            const bool stopped = interrupted.load(std::memory_order_relaxed);
            if (stopped && status.step != update.step) checkpoint(false);
            status.phase = stopped ? Phase::stopped : Phase::complete;
            if (!stopped) {
                model.save(model_path);
                models::publish(options.concept_key, model_path, source.fingerprint, update.step);
            }
            write_state(source.root, status);
            return status;
        } catch (...) {
            status.phase = Phase::failed;
            write_state(source.root, status);
            throw;
        }
    }
} // namespace genesia::training
