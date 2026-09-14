module;
#include "kernels.h"
#include <nlohmann/json.hpp>
module genesia.classifier.training;
import std;
import genesia.files;
import genesia.hash;
import genesia.dataset.operations;
namespace genesia::classifier {
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
        DeviceBuffer result(count * 4 + 4);
        GpuResult gpu{static_cast<float*>(result.data), reinterpret_cast<int*>(static_cast<std::byte*>(result.data) + count * 4)};
        nlohmann::json predictions = nlohmann::json::array();
        std::vector<std::vector<int>> confusion(count, std::vector<int>(count));
        double loss = 0;
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
            predictions.push_back({{"id", r.id}, {"label", r.label}, {"scores", std::vector<double>(scores.begin(), scores.end())}});
        }
        double precision = 0, recall = 0, f1 = 0;
        int correct              = 0;
        nlohmann::json per_class = nlohmann::json::array();
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
            per_class.push_back({{"name", model.classes[label]}, {"precision", p}, {"recall", r}, {"f1", f}, {"samples", actual}});
        }
        nlohmann::json metrics{{"step", step}, {"loss", loss / dataset.val.size()}, {"accuracy", double(correct) / dataset.val.size()}, {"precision", precision / count}, {"recall", recall / count}, {"f1", f1 / count}, {"classes", per_class}, {"confusion", confusion}, {"samples", dataset.val.size()}};
        files::write_json(output / std::format("step_{:04}_predictions.json", step), predictions);
        files::write_json(output / std::format("step_{:04}_metrics.json", step), metrics);
        return metrics;
    }
    nlohmann::json train(const TrainingOptions& options, const std::atomic_bool& interrupted, const std::function<void(const nlohmann::json&)>& progress, const std::function<void()>& yield) {
        const auto key_bytes = std::span{reinterpret_cast<const unsigned char*>(options.concept_key.data()), options.concept_key.size()};
        const dataset::Lock training_lock{"concept-" + sha256(key_bytes)};
        auto assigned = dataset::read_concept(options.concept_key);
        if (assigned.type == dataset::ConceptType::none) throw std::runtime_error{"Assign a concept type before training"};
        if (assigned.type == dataset::ConceptType::lora) throw std::runtime_error{"LoRA training is not implemented"};
        auto source = inspect(options.concept_key);
        {
            const dataset::Lock recovery{"image-moves"};
            dataset::recover_moves(source.root / ".genesia" / "classifier" / "audit-moves.json");
        }
        source = inspect(options.concept_key);
        if (!source.issue.empty()) throw std::runtime_error{source.issue};
        const auto root       = source.root / ".genesia" / "classifier";
        const auto manifest   = root / "training.json";
        const auto latest     = root / "latest.safetensors";
        const auto model_path = root / "publishing.safetensors";
        const auto snapshot   = std::filesystem::path{GENESIA_CACHE_DIRECTORY} / "classifier" / "data" / sha256(key_bytes) / source.fingerprint;
        nlohmann::json state;
        nlohmann::json saved_config = TrainingConfig{};
        int completed               = 0;
        if (!options.restart && !source.training.is_null()) {
            saved_config = source.training.at("config");
            if (source.training.at("fingerprint") != source.fingerprint) throw std::runtime_error{"Training data changed; Restart training is required"};
            if (std::filesystem::exists(latest)) {
                SafeFile saved(latest);
                const auto& meta = saved.header.at("__metadata__");
                if (meta.at("format") != "genesia-classifier-1") throw std::runtime_error{"Unsupported classifier checkpoint"};
                state        = nlohmann::json::parse(meta.at("training_state").get<std::string>());
                completed    = std::stoi(meta.at("step").get<std::string>());
                saved_config = state.at("config");
            } else throw std::runtime_error{"Training checkpoint is missing; Restart training is required"};
        }
        const bool resume     = !state.is_null();
        nlohmann::json config = options.restart ? nlohmann::json(TrainingConfig{}) : saved_config;
        for (auto item = options.overrides.begin(); item != options.overrides.end(); ++item) {
            if (!config.contains(item.key())) throw std::runtime_error{"Unknown training setting: " + item.key()};
            if (!options.restart && !source.training.is_null() && item.value() != saved_config.at(item.key())) throw std::runtime_error{"Changing " + item.key() + " requires Restart training"};
            config[item.key()] = item.value();
        }
        const TrainingConfig settings = config.get<TrainingConfig>();
        if (options.steps <= completed || options.steps <= 0) throw std::runtime_error{"Target steps must exceed the completed step"};
        if (!options.restart && !source.training.is_null() && options.steps < source.training.at("target").get<int>()) throw std::runtime_error{"Cumulative target steps cannot decrease"};
        if (settings.physical_batch <= 0 || settings.physical_batch > 256 || settings.effective_batch <= 0 || settings.eval_interval <= 0 || settings.save_interval <= 0 || settings.log_interval <= 0) throw std::runtime_error{"Invalid training batch or interval"};
        if (!std::filesystem::exists(snapshot / "dataset.json")) {
            const auto temporary = std::filesystem::path{snapshot.string() + ".building"};
            if (std::filesystem::exists(temporary)) std::filesystem::remove_all(temporary);
            try {
                prepare(source, temporary, interrupted, [&](std::size_t done, std::size_t total) { progress({{"stage", "preparing"}, {"completed", done}, {"total", total}}); });
                std::filesystem::rename(temporary, snapshot);
            } catch (...) {
                std::filesystem::remove_all(temporary);
                throw;
            }
        }
        if (interrupted.load()) throw std::runtime_error{"Training stopped before initialization"};
        if (options.restart && std::filesystem::exists(root)) {
            if (source.training.is_null()) throw std::runtime_error{"Cannot reset an unregistered concept state directory"};
            std::filesystem::remove_all(root);
            for (const auto& entry : std::filesystem::directory_iterator(snapshot.parent_path()))
                if (entry.path() != snapshot) std::filesystem::remove_all(entry.path());
            source.training = nullptr;
        }
        std::filesystem::create_directories(root);
        nlohmann::json status = source.training.is_null() ? nlohmann::json{{"version", 2}, {"published", nullptr}} : source.training;
        status["fingerprint"] = source.fingerprint;
        status["classes"]     = source.classes;
        status["state"]       = "preparing";
        status["target"]      = options.steps;
        status["step"]        = completed;
        status["config"]      = config;
        if (!assigned.locked) {
            assigned.locked = true;
            files::write_json(assigned.path / ".genesia" / "concept.json", assigned);
        }
        files::write_json(manifest, status);
        try {
            Dataset dataset(snapshot);
            files::write_json(root / "snapshot.json", dataset.metadata);
            if (!resume) state = {{"fingerprint", source.fingerprint}, {"schedule_steps", options.steps}, {"cursor", 0}};
            if (resume && state.at("split") != dataset.metadata) throw std::runtime_error{"Training split changed; Restart training is required"};
            state["config"]   = config;
            state["split"]    = dataset.metadata;
            const auto output = root / "metrics";
            std::filesystem::create_directories(output);
            Network model(resume ? latest : std::filesystem::path{u8"" CLASSIFIER_INITIAL_MODEL}, resume ? NetworkLoad::resume : NetworkLoad::pretrained);
            if (!resume) model.initialize_head(dataset.classes, settings.seed);
            model.metadata["format"] = "genesia-classifier-1";
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
                    progress({{"stage", "evaluation"}, {"metrics", metrics}});
                    files::write_json(output / "history.json", history);
                }
                model.save(latest, state);
                status["step"]  = update.step;
                status["state"] = "training";
                files::write_json(manifest, status);
            };
            if (!resume) checkpoint(true);
            yield();
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
                for (const auto& [shape, indices] : batch) {
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
                }
                cuda_check(cudaGraphLaunch(resources.optimizer, resources.stream));
                cuda_check(cudaMemcpyAsync(&update, model.optimizer->update, sizeof(update), cudaMemcpyDeviceToHost, resources.stream));
                cuda_check(cudaStreamSynchronize(resources.stream));
                double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                log << update.step << ',' << update.loss << ',' << std::sqrt(update.norm2) << ',' << seconds << '\n';
                log.flush();
                if (update.step % settings.log_interval == 0) {
                    progress({{"stage", "training"}, {"step", update.step}, {"target", options.steps}, {"loss", update.loss}, {"gradient_norm", std::sqrt(update.norm2)}, {"seconds", seconds}});
                }
                bool final      = update.step == options.steps || interrupted.load(std::memory_order_relaxed);
                bool evaluation = final || update.step % settings.eval_interval == 0;
                if (evaluation || update.step % settings.save_interval == 0) checkpoint(evaluation);
                yield();
            }
            const bool stopped = interrupted.load(std::memory_order_relaxed);
            if (stopped) checkpoint(false);
            status["state"] = stopped ? "stopped" : "complete";
            if (!stopped) {
                const auto current = inspect(options.concept_key);
                if (current.fingerprint != source.fingerprint) throw std::runtime_error{"Training data changed during training; Restart training is required"};
                model.save(model_path, state, false);
                const auto version        = files::digest(model_path);
                const auto published_path = root / "models" / (version + ".safetensors");
                std::filesystem::create_directories(published_path.parent_path());
                files::publish(model_path, published_path);
                status["published"] = {{"sha", version}, {"step", update.step}, {"fingerprint", source.fingerprint}};
            }
            files::write_json(manifest, status);
            if (!stopped && !source.training.is_null() && !source.training.at("published").is_null()) {
                const auto old = source.training.at("published").at("sha").get<std::string>();
                if (old != status.at("published").at("sha").get<std::string>()) std::filesystem::remove(root / "models" / (old + ".safetensors"));
            }
            return status;
        } catch (...) {
            status["state"] = interrupted.load() ? "stopped" : "failed";
            files::write_json(manifest, status);
            throw;
        }
    }
    nlohmann::json configuration() {
        return TrainingConfig{};
    }
} // namespace genesia::classifier
