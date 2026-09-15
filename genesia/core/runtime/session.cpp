module;
#include <genesia/cuda.h>
module genesia.runtime.session;
import std;
namespace genesia::runtime {
    Session::Session(generation::Visuals hooks, const bool browse) : visuals{std::move(hooks)}, loading{browse}, worker{[this, browse] { run(browse); }} {}
    Session::~Session() {
        shutdown();
        worker.join();
    }
    TaskStatus Session::submit(Request request) {
        TaskStatus initial;
        {
            const std::lock_guard lock{mutex};
            if (closing) throw std::runtime_error{"Genesia is closing"};
            if (!error.empty()) throw std::runtime_error{error};
            if (active || loading) throw std::runtime_error{"Finish the current operation first"};
            if (auto* generate = std::get_if<Generate>(&request.operation)) {
                auto& loras = generate->parameters.loras;
                std::erase_if(loras, [](const auto& lora) { return lora.weight == 0; });
                std::ranges::sort(loras, {}, &generation::Lora::concept_key);
                std::string previous;
                for (auto& lora : loras) {
                    if (!std::isfinite(lora.weight) || previous == lora.concept_key) throw std::runtime_error{"Invalid or repeated LoRA selection: " + lora.concept_key};
                    const auto model = models::resolve(lora.concept_key, dataset::ConceptType::lora);
                    lora.sha         = model.sha;
                    previous         = lora.concept_key;
                }
            }
            auto operation = std::make_shared<const Request>(std::move(request));
            const auto id  = next_id++;
            active         = TaskStatus{.id = id, .kind = Kind(operation->operation.index()), .started = std::chrono::steady_clock::now(), .request = operation};
            std::visit(
                [&]<typename T>(const T& value) {
                    if constexpr (std::same_as<T, Train>) active->concept_key = value.options.concept_key;
                    else if constexpr (std::same_as<T, Delete>) active->image_sha = value.sha;
                    else if constexpr (std::same_as<T, Normalize>) active->concept_key = value.root;
                    else if constexpr (!std::same_as<T, Generate> && !std::same_as<T, Mask>) active->concept_key = value.concept_key;
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
    void Session::observe(std::vector<Infer> requests, std::vector<dataset::File> masks, const bool include_generated) {
        std::string signature;
        for (const auto& request : requests) signature += request.descriptor->id + request.descriptor->sha + request.file->sha;
        for (const auto& image : masks) signature += "mask:" + image.sha;
        {
            const std::lock_guard lock{mutex};
            generated_masks = include_generated;
            if (signature == observed || closing) return;
            observed     = std::move(signature);
            wanted       = std::move(requests);
            wanted_masks = std::move(masks);
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
            if (kind == Kind::fix || kind == Kind::assign || kind == Kind::erase || kind == Kind::caption || kind == Kind::lora || (batch && batch->stage == Stage::moving)) return;
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
            wanted_masks.clear();
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
            result.idle     = !active && !loading && !inferring && wanted.empty() && wanted_masks.empty() && inspect_key.empty();
            result.finished = worker_done;
            result.pending  = !delivery.events.empty() || !delivery.previews.empty() || delivery.catalog.ready || !delivery.catalog.roots.empty() || !delivery.catalog.concepts.empty() || !delivery.catalog.classifiers.empty() || !delivery.catalog.captions.empty() || !delivery.catalog.concept_errors.empty();
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
            for (auto& [key, value] : update.captions) delivery.catalog.captions[key] = std::move(value);
            for (auto& [key, value] : update.loras) delivery.catalog.loras[key] = std::move(value);
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
                std::optional<dataset::File> mask;
                std::string selected;
                {
                    std::unique_lock lock{mutex};
                    condition.wait(lock, [this] { return closing || submitted || !inspect_key.empty() || !wanted.empty() || !wanted_masks.empty(); });
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
                    else if (!wanted_masks.empty()) {
                        mask = std::move(wanted_masks.front());
                        wanted_masks.erase(wanted_masks.begin());
                        inferring = true;
                    } else {
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
                } else if (mask) {
                    receive({EventKind::task, 0, {}, {}, segment({.file = std::move(mask)})});
                    const std::lock_guard lock{mutex};
                    inferring = false;
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
            foreground.network.reset();
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
                    for (int i = 0; i < operation.count; ++i) {
                        if (interrupted) throw Stopped{};
                        auto image = operation;
                        image.seed = operation.random_seed ? std::uniform_int_distribution<std::uint64_t>{}(random) : operation.seed + i;
                        emit(State::running, {BatchProgress{Stage::generating, std::size_t(i), std::size_t(operation.count)}});
                        const auto saved = engine->generate(catalog.index, id, image, interrupted);
                        if (!saved) throw Stopped{};
                        update_catalog(catalog.root("raw"));
                        std::vector<std::string> selected;
                        bool mask_saved;
                        {
                            const std::lock_guard lock{mutex};
                            selected   = activated;
                            mask_saved = generated_masks;
                        }
                        if (mask_saved && !interrupted) receive({EventKind::task, 0, {}, {}, segment({.file = saved->file}, saved->pixels)});
                        for (const auto& key : selected) {
                            if (interrupted) break;
                            receive({EventKind::task, 0, {}, {}, infer({.concept_key = key, .file = saved->file}, saved->pixels)});
                        }
                    }
                    emit(interrupted ? State::stopped : State::complete);
                } else if constexpr (std::same_as<T, Infer>) {
                    auto result = infer(operation);
                    emit(result.state, {}, std::move(result.result), std::move(result.error));
                } else if constexpr (std::same_as<T, Mask>) {
                    emit(State::running, {BatchProgress{Stage::segmenting, 0, 1}});
                    auto result = segment(operation);
                    if (interrupted) throw Stopped{};
                    if (result.state == State::complete && !operation.output.empty()) {
                        auto& mask = std::get<foreground::Result>(result.result.value);
                        std::filesystem::copy_file(mask.path, operation.output);
                        mask.path = operation.output;
                    }
                    emit(result.state, {BatchProgress{Stage::segmenting, 1, 1}}, std::move(result.result), std::move(result.error));
                } else if constexpr (std::same_as<T, Delete>) {
                    const auto root = files::path(operation.root);
                    if (root.has_parent_path() || operation.root.empty() || operation.root.front() == '.') throw std::runtime_error{"Delete requires a root dataset name"};
                    update_catalog(catalog.load(operation.root));
                    const auto result = dataset::delete_image(catalog.index, operation.root, operation.sha);
                    if (!result.paths.empty()) {
                        update_catalog(catalog.root(operation.root));
                        std::set<std::string> concepts;
                        for (const auto& path : result.paths) {
                            const auto relative = path.lexically_relative(project::directory / root);
                            if (std::distance(relative.begin(), relative.end()) > 1) concepts.insert(files::utf8(root / *relative.begin()));
                        }
                        for (const auto& key : concepts) update_catalog(catalog.describe(key, true));
                        const std::lock_guard lock{mutex};
                        std::erase_if(wanted, [&](const Infer& image) { return image.file && std::ranges::contains(result.paths, image.file->path); });
                        std::erase_if(wanted_masks, [&](const auto& image) { return std::ranges::contains(result.paths, image.path); });
                        observed.clear();
                    }
                    emit(result.error.empty() ? State::complete : State::failed, {}, {result}, result.error);
                } else if constexpr (std::same_as<T, Normalize>) {
                    const auto root = files::path(operation.root);
                    if (root.has_root_path() || root.has_parent_path() || operation.root.empty() || operation.root.front() == '.') throw std::runtime_error{"Normalize requires a root dataset name"};
                    emit(State::running, {BatchProgress{Stage::scanning}});
                    update_catalog(catalog.load(operation.root));
                    const auto result = dataset::normalize(catalog.index, operation.root, interrupted, [&](const dataset::NormalizeStage stage, const std::size_t completed, const std::size_t total) { progress({BatchProgress{stage == dataset::NormalizeStage::renaming ? Stage::moving : Stage::normalizing, completed, total}}); });
                    update_catalog(catalog.root(operation.root));
                    for (const auto& [key, info] : catalog.classifiers)
                        if (info.inspected && info.root.parent_path() == project::directory / root) update_catalog(catalog.describe(key, true));
                    for (const auto& collection : std::ranges::find(catalog.index.roots, operation.root, [](const dataset::Root& value) { return value.all.key; })->concepts)
                        if (dataset::read_concept(collection.key).type == dataset::ConceptType::lora) update_catalog(catalog.describe(collection.key, true));
                    {
                        const std::lock_guard lock{mutex};
                        wanted.clear();
                        wanted_masks.clear();
                        observed.clear();
                    }
                    emit(!result.error.empty() ? State::failed : result.stopped ? State::stopped : State::complete, {}, {result}, result.error);
                } else if constexpr (std::same_as<T, Classify>) {
                    const auto input    = std::filesystem::absolute(operation.input).lexically_normal();
                    const auto relative = input.lexically_relative(project::directory);
                    if (!relative.empty() && *relative.begin() != "..") update_catalog(catalog.load(files::utf8(*relative.begin())));
                    if (!predictions) predictions = std::make_unique<classification::Predictions>();
                    const auto result = classification::classify(catalog.index, operation.concept_key, input, *predictions, interrupted, progress);
                    moved(result.movement);
                    emit(State::complete, {}, {result});
                } else {
                    const auto key = [&] {
                        if constexpr (std::same_as<T, Train>) return operation.options.concept_key;
                        else return operation.concept_key;
                    }();
                    update_catalog(catalog.load(files::utf8(*files::path(key).begin())));
                    if constexpr (std::same_as<T, Assign>) {
                        const auto& root  = *std::ranges::find(catalog.index.roots, files::utf8(*files::path(key).begin()), [](const dataset::Root& value) { return value.all.key; });
                        const auto result = dataset::assign_type(root, key, operation.type);
                        update_catalog(catalog.describe(key));
                        emit(State::complete, {}, {result});
                    } else if constexpr (std::same_as<T, LoraModel>) {
                        const auto assigned = dataset::read_concept(key);
                        if (assigned.type != dataset::ConceptType::lora) throw std::runtime_error{"Expected a LoRA concept: " + key};
                        if (operation.remove) models::unpublish(key);
                        else if (!operation.input.empty()) models::import_lora(key, operation.input);
                        update_catalog(catalog.describe(key));
                        emit(State::complete, {}, {LoraModelResult{models::find(key)}});
                    } else if constexpr (std::same_as<T, Caption> || std::same_as<T, Export>) {
                        const auto assigned = dataset::read_concept(key);
                        if (assigned.type != dataset::ConceptType::lora) throw std::runtime_error{"Assign the LoRA type before managing captions or exporting: " + key};
                        const auto& root  = *std::ranges::find(catalog.index.roots, files::utf8(*files::path(key).begin()), [](const dataset::Root& value) { return value.all.key; });
                        const auto source = caption::inspect(assigned, root);
                        if constexpr (std::same_as<T, Caption>) {
                            const auto result = caption::edit(source, operation.folder, operation.tags, operation.bypass);
                            if (operation.tags || operation.bypass) update_catalog(catalog.describe(key));
                            emit(State::complete, {}, {result});
                        } else {
                            const auto result = caption::export_dataset(source, root, operation.output, interrupted, [&](const dataset::File& image) { return foreground.infer(image).path; }, [&](const std::size_t completed, const std::size_t total) { progress({BatchProgress{Stage::exporting, completed, total}}); });
                            emit(State::complete, {}, {result});
                        }
                    } else {
                        const auto source = catalog.inspect(key);
                        if constexpr (std::same_as<T, Train>) {
                            release_generation();
                            predictions.reset();
                            foreground.network.reset();
                            const auto result = training::train(operation.options, source, interrupted, progress);
                            update_catalog(catalog.describe(key));
                            {
                                const std::lock_guard lock{mutex};
                                wanted.clear();
                                wanted_masks.clear();
                                observed.clear();
                            }
                            emit(result.phase == training::Phase::stopped ? State::stopped : State::complete, {}, {result});
                        } else if constexpr (std::same_as<T, Audit>) {
                            if (!predictions) predictions = std::make_unique<classification::Predictions>();
                            emit(State::complete, {}, {classification::audit(source, *predictions, operation.refresh, interrupted, progress)});
                        } else {
                            const auto result = classification::fix(source, operation.sha, operation.category);
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
                request.descriptor = loaded == predictions->pipeline.models.end() ? models::resolve(request.concept_key, dataset::ConceptType::classifier) : loaded->descriptor;
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
    TaskStatus Session::segment(Mask request, const std::span<const std::uint8_t> rgb) {
        TaskStatus result{.kind = Kind::mask, .state = State::complete, .request = std::make_shared<const Request>(Request{request})};
        try {
            if (!request.file) request.file = catalog.index.identify(request.input);
            result.image_sha    = request.file->sha;
            auto mask           = foreground.infer(*request.file, rgb);
            result.model_sha    = mask.model_sha;
            result.result.value = std::move(mask);
        } catch (const std::exception& failure) {
            result.state = State::failed;
            result.error = failure.what();
        }
        return result;
    }
} // namespace genesia::runtime
