module;
#include <genesia/cuda.h>
module genesia.runtime.session;
import genesia.io.hash;
import genesia.compute.device;
import std;
namespace genesia::runtime {
    Session::Session(generation::Visuals hooks) : visuals{std::move(hooks)}, worker{[this] { run(); }} {}
    Session::~Session() {
        shutdown();
        worker.join();
    }
    std::uint64_t Session::enqueue(Request request) {
        Task task;
        task.request    = std::move(request);
        const auto kind = Kind(task.request.operation.index());
        std::string concept_key;
        std::visit(
            [&]<typename T>(const T& operation) {
                if constexpr (std::same_as<T, Train>) concept_key = operation.options.concept_key;
                else if constexpr (!std::same_as<T, Generate>) concept_key = operation.concept_key;
            },
            task.request.operation);
        if (kind == Kind::train) {
            concept_key                                                 = files::utf8(files::path(concept_key));
            std::get<Train>(task.request.operation).options.concept_key = concept_key;
            task.type_lease                                             = std::make_shared<files::Lock>("concept-type-" + sha256({reinterpret_cast<const unsigned char*>(concept_key.data()), concept_key.size()}), false, project::state_directory, true);
            if (!task.type_lease->acquired) throw std::runtime_error{"Concept type is being changed: " + concept_key};
            const auto source = dataset::read_concept(concept_key);
            if (source.type == dataset::ConceptType::none) throw std::runtime_error{"Assign a concept type before training"};
            if (source.type == dataset::ConceptType::lora) throw std::runtime_error{"LoRA training is not implemented"};
        }
        std::shared_ptr<generation::Engine> engine;
        std::uint64_t id;
        {
            const std::lock_guard lock{mutex};
            if (closing) throw std::runtime_error{"Task session is closing"};
            if (!error.empty()) throw std::runtime_error{error};
            task.id = id = next_id++;
            TaskStatus status{.id = id, .kind = kind, .concept_key = std::move(concept_key), .request = task.request};
            if (const auto* infer = std::get_if<Infer>(&task.request.operation)) {
                if (infer->descriptor) status.model_sha = infer->descriptor->sha;
                if (infer->file) status.image_sha = infer->file->sha;
                std::get<Infer>(status.request.operation).descriptor.reset();
            }
            jobs[id] = status;
            events.push_back({EventKind::task, id, {}, {}, std::move(status)});
            if (kind == Kind::infer) {
                immediate.push_back(std::move(task));
                engine = generation;
            } else queue.push_back(std::move(task));
        }
        if (engine) engine->request_yield();
        condition.notify_all();
        if (visuals.notify) visuals.notify();
        return id;
    }
    void Session::observe(std::vector<Infer> requests) {
        std::string signature;
        for (const auto& request : requests) signature += request.descriptor->id + request.descriptor->sha + request.file->sha;
        {
            const std::lock_guard lock{mutex};
            if (signature == observed || closing) return;
            observed = std::move(signature);
            std::erase_if(immediate, [&](const Task& task) {
                if (!task.request.transient) return false;
                jobs.erase(task.id);
                return true;
            });
            if (active && active->request.transient) active->interrupted->store(true);
        }
        for (auto& request : requests) enqueue({std::move(request), true});
    }
    void Session::activate(std::vector<std::string> models) {
        const std::lock_guard lock{mutex};
        activated = std::move(models);
    }
    void Session::configure_preview(const bool enabled, const bool visible) {
        std::shared_ptr<generation::Engine> engine;
        {
            const std::lock_guard lock{mutex};
            preview_enabled = enabled;
            preview_visible = visible;
            engine          = generation;
        }
        if (engine) engine->configure(enabled, visible);
    }
    void Session::cancel(const std::uint64_t id) {
        std::optional<Task> removed;
        std::shared_ptr<generation::Engine> engine;
        {
            const std::lock_guard lock{mutex};
            for (auto* current : {&active, &suspended}) {
                if (!*current || (*current)->id != id) continue;
                const auto& status = jobs.at(id);
                if (status.state != State::running && status.state != State::queued) return;
                if (status.kind == Kind::fix || status.kind == Kind::undo || status.kind == Kind::assign) return;
                const auto* batch = std::get_if<BatchProgress>(&status.progress.value);
                if (batch && batch->stage == Stage::moving) return;
                (*current)->interrupted->store(true);
                jobs.at(id).stopping = true;
                if (status.kind == Kind::generate) engine = generation;
            }
            for (auto* pending : {&queue, &immediate}) {
                const auto found = std::ranges::find(*pending, id, &Task::id);
                if (found == pending->end()) continue;
                removed = std::move(*found);
                pending->erase(found);
                break;
            }
        }
        if (engine) engine->cancel();
        if (removed) emit(*removed, State::stopped);
    }
    void Session::shutdown() {
        std::shared_ptr<generation::Engine> engine;
        {
            const std::lock_guard lock{mutex};
            closing = true;
            for (auto* current : {&active, &suspended})
                if (*current) (*current)->interrupted->store(true);
            for (auto* pending : {&queue, &immediate}) {
                for (const auto& task : *pending) {
                    auto& job = jobs.at(task.id);
                    job.state = State::stopped;
                    events.push_back({EventKind::task, task.id, {}, {}, job});
                }
                pending->clear();
            }
            engine = generation;
        }
        if (engine) engine->cancel();
        condition.notify_all();
    }
    Snapshot Session::snapshot() {
        Snapshot result;
        std::shared_ptr<generation::Engine> engine;
        {
            const std::lock_guard lock{mutex};
            if (active) result.active = jobs.at(active->id);
            result.jobs     = jobs;
            result.idle     = !active && queue.empty() && immediate.empty() && std::ranges::all_of(jobs, [](const auto& entry) { return entry.second.state >= State::complete; });
            result.finished = worker_done;
            result.pending  = !events.empty() || !previews.empty();
            result.error    = error;
            engine          = generation;
        }
        if (engine) result.generation = engine->observe();
        return result;
    }
    Delivery Session::drain() {
        const std::lock_guard lock{mutex};
        return {std::exchange(events, {}), std::exchange(previews, {})};
    }
    void Session::emit(const Task& task, const State state, Progress progress, Result result, std::string failure) {
        {
            const std::lock_guard lock{mutex};
            auto& status = jobs.at(task.id);
            if (status.state == State::queued && state == State::running) status.started = std::chrono::steady_clock::now();
            status.state = state;
            if (progress.value.index()) status.progress = std::move(progress);
            if (result.value.index()) status.result = std::move(result);
            status.error = std::move(failure);
            events.push_back({EventKind::task, task.id, {}, {}, status});
        }
        if (visuals.notify) visuals.notify();
    }
    void Session::receive(Event event) {
        std::vector<std::string> selected;
        const auto path = event.record.path;
        {
            const std::lock_guard lock{mutex};
            if (event.kind == EventKind::task) {
                auto& status  = jobs.at(event.id);
                status.state  = event.task.state;
                status.result = std::move(event.task.result);
                status.error  = std::move(event.task.error);
                event.task    = status;
            } else if (event.kind == EventKind::saved && !closing) selected = activated;
            events.push_back(std::move(event));
        }
        for (const auto& key : selected) {
            try {
                enqueue({Infer{.concept_key = key, .input = path}});
            } catch (const std::exception& failure) {
                std::println(std::cerr, "Post-save inference: {}", failure.what());
            }
        }
        if (visuals.notify) visuals.notify();
    }
    void Session::release_generation() {
        std::shared_ptr<generation::Engine> previous;
        {
            const std::lock_guard lock{mutex};
            previous = std::exchange(generation, {});
        }
        if (previous) previous->finish();
        previous.reset();
    }
    void Session::run() {
        std::unique_ptr<files::Lock> lease;
        std::string gpu_name;
        const auto gpu_locks = std::filesystem::temp_directory_path() / "genesia-gpu";
        const auto release   = [&] {
            release_generation();
            predictions.reset();
            lease.reset();
        };
        try {
            for (;;) {
                Task task;
                {
                    std::unique_lock lock{mutex};
                    while (!closing && immediate.empty() && queue.empty()) {
                        lock.unlock();
                        if (lease) {
                            const files::Lock demand{gpu_name + "-wanted", false, gpu_locks};
                            if (!demand.acquired) {
                                release();
                                for (;;) {
                                    const files::Lock waiting{gpu_name + "-wanted", false, gpu_locks};
                                    if (waiting.acquired) break;
                                    std::unique_lock lock{mutex};
                                    if (closing) break;
                                    condition.wait_for(lock, std::chrono::milliseconds{100});
                                }
                            }
                        }
                        lock.lock();
                        condition.wait_for(lock, std::chrono::milliseconds{100}, [this] { return closing || !queue.empty() || !immediate.empty(); });
                    }
                    if (closing) break;
                    auto& pending = immediate.empty() ? queue : immediate;
                    task          = std::move(pending.front());
                    pending.pop_front();
                    active = task;
                }
                emit(task, State::running);
                try {
                    if (!predictions) predictions = std::make_unique<classification::Predictions>();
                    bool gpu = std::visit(
                        [&]<typename T>(T& operation) {
                            if constexpr (std::same_as<T, Infer>) {
                                if (!operation.descriptor) operation.descriptor = models::resolve(operation.concept_key);
                                if (!operation.file) {
                                    dataset::Index index;
                                    operation.file = index.identify(operation.input);
                                }
                                const std::lock_guard lock{mutex};
                                auto& status     = jobs.at(task.id);
                                status.model_sha = operation.descriptor->sha;
                                status.image_sha = operation.file->sha;
                                return operation.refresh || !predictions->cache.find(*operation.descriptor, operation.file->sha);
                            } else if constexpr (std::same_as<T, Audit>) {
                                const auto source = training::inspect(operation.concept_key);
                                return operation.refresh || !classification::view(source, predictions->cache).complete;
                            } else return std::same_as<T, Generate> || std::same_as<T, Train> || std::same_as<T, Classify>;
                        },
                        task.request.operation);
                    if (gpu && gpu_name.empty()) {
                        cudaDeviceProp properties;
                        compute::check(cudaGetDeviceProperties(&properties, 0));
                        gpu_name = "gpu-";
                        for (const auto byte : properties.uuid.bytes) gpu_name += std::format("{:02x}", static_cast<unsigned char>(byte));
                    }
                    std::unique_ptr<files::Lock> demand;
                    while (gpu && !lease) {
                        auto attempt = std::make_unique<files::Lock>(gpu_name, false, gpu_locks);
                        if (attempt->acquired) {
                            lease = std::move(attempt);
                            break;
                        }
                        if (task.interrupted->load()) throw Stopped{};
                        if (!demand) {
                            auto wanted = std::make_unique<files::Lock>(gpu_name + "-wanted", false, gpu_locks);
                            if (wanted->acquired) demand = std::move(wanted);
                        }
                        std::unique_lock lock{mutex};
                        condition.wait_for(lock, std::chrono::milliseconds{100});
                    }
                    demand.reset();
                    execute(task);
                } catch (const Stopped&) {
                    emit(task, State::stopped);
                } catch (const std::exception& failure) {
                    emit(task, State::failed, {}, {}, failure.what());
                }
                {
                    const std::lock_guard lock{mutex};
                    active.reset();
                    if (task.request.transient) jobs.erase(task.id);
                }
                if (lease) {
                    const files::Lock demand{gpu_name + "-wanted", false, gpu_locks};
                    if (!demand.acquired) {
                        release();
                        for (;;) {
                            const files::Lock waiting{gpu_name + "-wanted", false, gpu_locks};
                            if (waiting.acquired) break;
                            std::unique_lock lock{mutex};
                            if (closing) break;
                            condition.wait_for(lock, std::chrono::milliseconds{100});
                        }
                    }
                }
                if (visuals.notify) visuals.notify();
            }
            release();
        } catch (const std::exception& failure) {
            const std::lock_guard lock{mutex};
            error = failure.what();
            queue.clear();
            immediate.clear();
        }
        {
            const std::lock_guard lock{mutex};
            worker_done = true;
        }
        if (visuals.notify) visuals.notify();
    }
    void Session::execute(const Task& task) {
        if (task.interrupted->load()) throw Stopped{};
        const auto progress    = [&](const Progress& value) { emit(task, State::running, value); };
        const auto cooperative = [this] { yield(); };
        std::visit(
            [&]<typename T>(const T& operation) {
                if constexpr (std::same_as<T, Generate>) {
                    std::shared_ptr<generation::Engine> engine;
                    {
                        const std::lock_guard lock{mutex};
                        engine = generation;
                    }
                    if (!engine) {
                        engine = std::make_shared<generation::Engine>(
                            visuals, [this](Event event) { receive(std::move(event)); },
                            [this](PreviewFrame frame) {
                                {
                                    const std::lock_guard lock{mutex};
                                    previews.push_back(std::move(frame));
                                }
                                if (visuals.notify) visuals.notify();
                            });
                        const std::lock_guard lock{mutex};
                        engine->configure(preview_enabled, preview_visible);
                        generation = engine;
                        if (!immediate.empty()) engine->request_yield();
                    }
                    if (!engine->generate(task.id, operation, *task.interrupted, cooperative)) emit(task, State::stopped);
                } else if constexpr (std::same_as<T, Train>) {
                    release_generation();
                    predictions.reset();
                    predictions       = std::make_unique<classification::Predictions>();
                    const auto result = training::train(operation.options, *task.interrupted, progress, cooperative);
                    emit(task, result.phase == training::Phase::stopped ? State::stopped : State::complete, {}, {result});
                } else if constexpr (std::same_as<T, Infer>) {
                    const auto descriptor = operation.descriptor ? *operation.descriptor : models::resolve(operation.concept_key);
                    dataset::Index index;
                    const auto file = operation.file ? *operation.file : index.identify(operation.input);
                    {
                        const std::lock_guard lock{mutex};
                        jobs.at(task.id).model_sha = descriptor.sha;
                        jobs.at(task.id).image_sha = file.sha;
                    }
                    std::vector<std::string> selected;
                    {
                        const std::lock_guard lock{mutex};
                        selected = activated;
                    }
                    predictions->active = std::move(selected);
                    auto result         = predictions->infer(descriptor, file, operation.refresh);
                    if (task.interrupted->load()) throw Stopped{};
                    emit(task, State::complete, {}, {std::move(result)});
                } else if constexpr (std::same_as<T, Audit>) emit(task, State::complete, {}, {classification::audit(operation.concept_key, *predictions, operation.refresh, *task.interrupted, progress, cooperative)});
                else if constexpr (std::same_as<T, Fix>) emit(task, State::complete, {}, {classification::fix(operation.concept_key, operation.sha, operation.category)});
                else if constexpr (std::same_as<T, Undo>) emit(task, State::complete, {}, {classification::undo(operation.concept_key)});
                else if constexpr (std::same_as<T, Classify>) emit(task, State::complete, {}, {classification::classify(operation.concept_key, operation.input, *predictions, *task.interrupted, progress, cooperative)});
                else if constexpr (std::same_as<T, Assign>) emit(task, State::complete, {}, {dataset::assign_type(operation.concept_key, operation.type)});
            },
            task.request.operation);
    }
    void Session::yield() {
        Task task;
        {
            const std::lock_guard lock{mutex};
            if (immediate.empty() || closing) return;
            task = std::move(immediate.front());
            immediate.pop_front();
            suspended = active;
            active    = task;
        }
        emit(task, State::running);
        try {
            execute(task);
        } catch (const Stopped&) {
            emit(task, State::stopped);
        } catch (const std::exception& failure) {
            emit(task, State::failed, {}, {}, failure.what());
        }
        std::shared_ptr<generation::Engine> engine;
        {
            const std::lock_guard lock{mutex};
            active = std::exchange(suspended, {});
            if (task.request.transient) jobs.erase(task.id);
            if (!immediate.empty()) engine = generation;
        }
        if (engine) engine->request_yield();
    }
} // namespace genesia::runtime
