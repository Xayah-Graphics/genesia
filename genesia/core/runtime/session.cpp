module;
#include <genesia/cuda.h>
module genesia.runtime.session;
import genesia.io.hash;
import std;
namespace genesia::runtime {
    Session::Session(generation::Visuals hooks, const bool browse) : visuals{std::move(hooks)}, loading{browse}, worker{[this, browse] { run(browse); }} {}
    Session::~Session() {
        shutdown();
        worker.join();
    }
    TaskStatus Session::submit(Request request) {
        auto operation = std::make_shared<const Request>(std::move(request));
        TaskStatus initial;
        {
            const std::lock_guard lock{mutex};
            if (closing) throw std::runtime_error{"Genesia is closing"};
            if (!error.empty()) throw std::runtime_error{error};
            if (active || loading) throw std::runtime_error{"Finish the current operation first"};
            const auto id = next_id++;
            active        = TaskStatus{.id = id, .kind = Kind(operation->operation.index()), .started = std::chrono::steady_clock::now(), .request = operation};
            std::visit(
                [&]<typename T>(const T& value) {
                    if constexpr (std::same_as<T, Train>) active->concept_key = value.options.concept_key;
                    else if constexpr (!std::same_as<T, Generate>) active->concept_key = value.concept_key;
                },
                operation->operation);
            interrupted = false;
            submitted   = true;
            initial     = *active;
            delivery.events.push_back({EventKind::task, id, {}, {}, initial});
        }
        condition.notify_one();
        if (visuals.notify) visuals.notify();
        return initial;
    }
    void Session::select(std::string key) {
        {
            const std::lock_guard lock{mutex};
            if (selection == key) return;
            selection   = key;
            inspect_key = std::move(key);
        }
        condition.notify_one();
    }
    void Session::observe(std::vector<Infer> requests) {
        std::string signature;
        for (const auto& request : requests) signature += request.descriptor->id + request.descriptor->sha + request.file->sha;
        {
            const std::lock_guard lock{mutex};
            if (signature == observed || closing) return;
            observed = std::move(signature);
            wanted   = std::move(requests);
        }
        condition.notify_one();
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
        std::shared_ptr<generation::Engine> engine;
        {
            const std::lock_guard lock{mutex};
            if (!active || active->id != id) return;
            const auto kind   = active->kind;
            const auto* batch = std::get_if<BatchProgress>(&active->progress.value);
            if (kind == Kind::fix || kind == Kind::undo || kind == Kind::assign || (batch && batch->stage == Stage::moving)) return;
            interrupted      = true;
            active->stopping = true;
            if (kind == Kind::generate) engine = generation;
        }
        if (engine) engine->cancel();
    }
    void Session::shutdown() {
        std::shared_ptr<generation::Engine> engine;
        {
            const std::lock_guard lock{mutex};
            closing     = true;
            interrupted = true;
            wanted.clear();
            inspect_key.clear();
            engine = generation;
        }
        if (engine) engine->cancel();
        condition.notify_one();
    }
    Snapshot Session::snapshot() {
        Snapshot result;
        std::shared_ptr<generation::Engine> engine;
        {
            const std::lock_guard lock{mutex};
            result.active   = active;
            result.idle     = !active && !loading && !inferring && wanted.empty() && inspect_key.empty();
            result.finished = worker_done;
            result.pending  = !delivery.events.empty() || !delivery.previews.empty() || delivery.catalog.ready || !delivery.catalog.roots.empty() || !delivery.catalog.concepts.empty() || !delivery.catalog.classifiers.empty() || !delivery.catalog.concept_errors.empty();
            result.error    = error;
            engine          = generation;
        }
        if (engine) result.generation = engine->observe();
        return result;
    }
    Delivery Session::drain() {
        const std::lock_guard lock{mutex};
        return std::exchange(delivery, {});
    }
    void Session::emit(const State state, Progress progress, Result result, std::string failure) {
        {
            const std::lock_guard lock{mutex};
            active->state = state;
            if (progress.value.index()) active->progress = std::move(progress);
            auto event   = *active;
            event.result = std::move(result);
            event.error  = std::move(failure);
            delivery.events.push_back({EventKind::task, event.id, {}, {}, std::move(event)});
            if (state >= State::complete) active.reset();
        }
        if (visuals.notify) visuals.notify();
    }
    void Session::receive(Event event) {
        {
            const std::lock_guard lock{mutex};
            if (event.kind == EventKind::task && event.id) {
                active->state = event.task.state;
                event.task    = *active;
            }
            delivery.events.push_back(std::move(event));
        }
        if (visuals.notify) visuals.notify();
    }
    void Session::update_catalog(CatalogState update) {
        {
            const std::lock_guard lock{mutex};
            for (auto& root : update.roots) {
                const auto found = std::ranges::find(delivery.catalog.roots, root.all.key, [](const dataset::Root& item) { return item.all.key; });
                if (found == delivery.catalog.roots.end()) delivery.catalog.roots.push_back(std::move(root));
                else *found = std::move(root);
            }
            for (auto& [key, value] : update.concepts) delivery.catalog.concepts[key] = std::move(value);
            for (auto& [key, value] : update.classifiers) delivery.catalog.classifiers[key] = std::move(value);
            for (auto& [key, value] : update.concept_errors) delivery.catalog.concept_errors[key] = std::move(value);
            delivery.catalog.ready |= update.ready;
        }
        if (visuals.notify) visuals.notify();
    }
    void Session::moved(const dataset::MoveResult& movement) {
        for (const auto& root : catalog.index.apply(movement.paths)) update_catalog(catalog.root(root));
        std::set<std::string> concepts;
        for (const auto& move : movement.paths)
            for (const auto& path : {move.source, move.destination}) {
                const auto relative = path.lexically_relative(project::directory);
                if (relative.empty() || *relative.begin() == ".." || std::distance(relative.begin(), relative.end()) < 3) continue;
                auto part       = relative.begin();
                const auto root = *part++;
                concepts.insert(files::utf8(root / *part));
            }
        for (const auto& key : concepts) update_catalog(catalog.describe(key, true));
    }
    void Session::release_generation() {
        std::shared_ptr<generation::Engine> previous;
        {
            const std::lock_guard lock{mutex};
            previous = std::exchange(generation, {});
        }
        if (previous) previous->finish();
    }
    void Session::run(const bool browse) {
        try {
            if (browse) update_catalog(catalog.load());
            {
                const std::lock_guard lock{mutex};
                loading = false;
            }
            if (visuals.notify) visuals.notify();
            for (;;) {
                std::shared_ptr<const Request> request;
                std::optional<Infer> image;
                std::string selected;
                {
                    std::unique_lock lock{mutex};
                    condition.wait(lock, [this] { return closing || submitted || !inspect_key.empty() || !wanted.empty(); });
                    if (closing) {
                        const bool pending = active.has_value();
                        lock.unlock();
                        if (pending) emit(State::stopped);
                        break;
                    }
                    if (submitted) {
                        request   = active->request;
                        submitted = false;
                    } else if (!inspect_key.empty()) selected = std::exchange(inspect_key, {});
                    else {
                        image = std::move(wanted.front());
                        wanted.erase(wanted.begin());
                        inferring = true;
                    }
                }
                if (!selected.empty()) {
                    try {
                        update_catalog(catalog.load(files::utf8(*files::path(selected).begin())));
                        catalog.inspect(selected);
                        update_catalog(catalog.describe(selected));
                    } catch (const std::exception& failure) {
                        CatalogState update;
                        update.concept_errors[selected] = failure.what();
                        update_catalog(std::move(update));
                    }
                } else if (image) {
                    receive({EventKind::task, 0, {}, {}, infer(std::move(*image))});
                    const std::lock_guard lock{mutex};
                    inferring = false;
                } else if (request) {
                    try {
                        if (interrupted) throw Stopped{};
                        execute(*request);
                    } catch (const Stopped&) {
                        emit(State::stopped);
                    } catch (const std::exception& failure) {
                        if (const auto* train = std::get_if<Train>(&request->operation)) update_catalog(catalog.describe(train->options.concept_key));
                        emit(State::failed, {}, {}, failure.what());
                    }
                }
            }
            release_generation();
            predictions.reset();
            catalog.index.flush();
        } catch (const std::exception& failure) {
            const std::lock_guard lock{mutex};
            error = failure.what();
        }
        {
            const std::lock_guard lock{mutex};
            worker_done = true;
        }
        if (visuals.notify) visuals.notify();
    }
    void Session::execute(const Request& request) {
        const auto progress = [&](const Progress& value) { emit(State::running, value); };
        std::visit(
            [&]<typename T>(const T& operation) {
                if constexpr (std::same_as<T, Generate>) {
                    update_catalog(catalog.load(std::string{"raw"}));
                    std::shared_ptr<generation::Engine> engine;
                    std::uint64_t id;
                    {
                        const std::lock_guard lock{mutex};
                        engine = generation;
                        id     = active->id;
                    }
                    if (!engine) {
                        engine = std::make_shared<generation::Engine>(
                            visuals, [this](Event event) { receive(std::move(event)); },
                            [this](PreviewFrame frame) {
                                {
                                    const std::lock_guard lock{mutex};
                                    delivery.previews.push_back(std::move(frame));
                                }
                                if (visuals.notify) visuals.notify();
                            });
                        const std::lock_guard lock{mutex};
                        engine->configure(preview_enabled, preview_visible);
                        generation = engine;
                    }
                    std::random_device random;
                    Generated result;
                    for (int i = 0; i < operation.count; ++i) {
                        if (interrupted) throw Stopped{};
                        auto image = operation;
                        image.seed = operation.random_seed ? std::uniform_int_distribution<std::uint64_t>{}(random) : operation.seed + i;
                        emit(State::running, {BatchProgress{Stage::generating, std::size_t(i), std::size_t(operation.count)}});
                        const auto saved = engine->generate(catalog.index, id, image, interrupted);
                        if (!saved) throw Stopped{};
                        update_catalog(catalog.root("raw"));
                        result = {saved->file.path, saved->record.seed};
                        std::vector<std::string> selected;
                        {
                            const std::lock_guard lock{mutex};
                            selected = activated;
                        }
                        for (const auto& key : selected) {
                            if (interrupted) break;
                            receive({EventKind::task, 0, {}, {}, infer({.concept_key = key, .file = saved->file}, saved->pixels)});
                        }
                    }
                    emit(interrupted ? State::stopped : State::complete, {}, {result});
                } else if constexpr (std::same_as<T, Infer>) {
                    auto result = infer(operation);
                    emit(result.state, {}, std::move(result.result), std::move(result.error));
                } else if constexpr (std::same_as<T, Classify>) {
                    const auto input   = std::filesystem::absolute(operation.input).lexically_normal();
                    const auto text    = files::utf8(input);
                    const auto journal = project::state_directory / "operations" / (sha256({reinterpret_cast<const unsigned char*>(text.data()), text.size()}) + ".json");
                    moved(dataset::recover_moves(journal));
                    const auto relative = input.lexically_relative(project::directory);
                    if (!relative.empty() && *relative.begin() != "..") update_catalog(catalog.load(files::utf8(*relative.begin())));
                    if (!predictions) predictions = std::make_unique<classification::Predictions>();
                    const auto result = classification::classify(catalog.index, operation.concept_key, input, journal, *predictions, interrupted, progress);
                    moved(result.movement);
                    emit(State::complete, {}, {result});
                } else {
                    const auto key = [&] {
                        if constexpr (std::same_as<T, Train>) return operation.options.concept_key;
                        else return operation.concept_key;
                    }();
                    update_catalog(catalog.load(files::utf8(*files::path(key).begin())));
                    if constexpr (std::same_as<T, Assign>) {
                        const auto result = dataset::assign_type(key, operation.type);
                        update_catalog(catalog.describe(key));
                        emit(State::complete, {}, {result});
                    } else {
                        const auto source = catalog.inspect(key);
                        if constexpr (std::same_as<T, Train>) {
                            release_generation();
                            predictions.reset();
                            const auto result = training::train(operation.options, source, interrupted, progress);
                            update_catalog(catalog.describe(key));
                            {
                                const std::lock_guard lock{mutex};
                                wanted.clear();
                                observed.clear();
                            }
                            emit(result.phase == training::Phase::stopped ? State::stopped : State::complete, {}, {result});
                        } else if constexpr (std::same_as<T, Audit>) {
                            if (!predictions) predictions = std::make_unique<classification::Predictions>();
                            emit(State::complete, {}, {classification::audit(source, *predictions, operation.refresh, interrupted, progress)});
                        } else {
                            const auto result = [&] {
                                if constexpr (std::same_as<T, Fix>) return classification::fix(source, operation.sha, operation.category);
                                else return classification::undo(source);
                            }();
                            moved(result);
                            emit(State::complete, {}, {result});
                        }
                    }
                }
            },
            request.operation);
    }
    TaskStatus Session::infer(Infer request, const std::span<const std::uint8_t> rgb) {
        TaskStatus result{.kind = Kind::infer, .concept_key = request.concept_key, .state = State::complete, .request = std::make_shared<const Request>(Request{request})};
        try {
            if (!predictions) predictions = std::make_unique<classification::Predictions>();
            if (!request.descriptor) {
                const auto loaded  = std::ranges::find(predictions->pipeline.models, request.concept_key, [](const classification::LoadedClassifier& model) { return model.descriptor.id; });
                request.descriptor = loaded == predictions->pipeline.models.end() ? models::resolve(request.concept_key) : loaded->descriptor;
            }
            if (!request.file) request.file = catalog.index.identify(request.input);
            result.model_sha = request.descriptor->sha;
            result.image_sha = request.file->sha;
            {
                const std::lock_guard lock{mutex};
                predictions->active = activated;
            }
            result.result.value = predictions->infer(*request.descriptor, *request.file, request.refresh, rgb);
        } catch (const std::exception& failure) {
            result.state = State::failed;
            result.error = failure.what();
        }
        return result;
    }
} // namespace genesia::runtime
