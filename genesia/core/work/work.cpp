module;
#include "../sdxl/control.h"
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>
module genesia.work;
import genesia.neural.inference_runtime;
import genesia.images;
import genesia.files;
import genesia.hash;
import std;
namespace genesia::work {
    Session::Session(Visuals hooks) : visuals{std::move(hooks)}, stream{::cuda::devices[0]}, control{stream, ::cuda::pinned_default_memory_pool(), 1, ::cuda::no_init} {
        std::construct_at(control.data());
        try {
            if (visuals.publish) {
                int least, greatest;
                neural::check(cudaDeviceGetStreamPriorityRange(&least, &greatest));
                preview_stream = ::cuda::stream{::cuda::devices[0], least};
                neural::check(cudaEventCreateWithFlags(std::out_ptr(preview_finished), cudaEventDisableTiming));
                preview_worker = std::jthread{[this] { preview_images(); }};
            }
            io     = std::jthread{[this] { write_files(); }};
            worker = std::jthread{[this] { run(); }};
        } catch (...) {
            {
                const std::lock_guard lock{mutex};
                io_closing = preview_closing = true;
            }
            condition.notify_all();
            if (io.joinable()) io.join();
            if (preview_worker.joinable()) preview_worker.join();
            throw;
        }
    }
    Session::~Session() {
        shutdown();
        worker.join();
        {
            const std::lock_guard lock{mutex};
            io_closing = preview_closing = true;
        }
        condition.notify_all();
        io.join();
        if (preview_worker.joinable()) preview_worker.join();
        inference.reset();
        model.reset();
        if (visuals.publish) preview_stream.sync();
        stream.sync();
    }
    std::uint64_t Session::enqueue(Request request) {
        if (request.kind == Kind::train) {
            request.concept_key          = files::utf8(files::path(request.concept_key));
            request.training.concept_key = request.concept_key;
            const auto key               = std::span{reinterpret_cast<const unsigned char*>(request.concept_key.data()), request.concept_key.size()};
            request.type_lease           = std::make_shared<dataset::Lock>("concept-type-" + sha256(key), false, dataset::state_directory, true);
            if (!request.type_lease->acquired) throw std::runtime_error{"Concept type is being changed: " + request.concept_key};
            const auto assigned = dataset::read_concept(request.concept_key);
            if (assigned.type == dataset::ConceptType::none) throw std::runtime_error{"Assign a concept type before training"};
            if (assigned.type == dataset::ConceptType::lora) throw std::runtime_error{"LoRA training is not implemented"};
        }
        std::uint64_t id;
        {
            const std::lock_guard lock{mutex};
            if (closing) throw std::runtime_error{"Task session is closing"};
            if (!error.empty()) throw std::runtime_error{error};
            request.id = id = next_id++;
            jobs[id]        = {{"id", id}, {"kind", static_cast<int>(request.kind)}, {"concept", request.concept_key}, {"state", "queued"}};
            auto& job       = jobs.at(id);
            if (request.kind == Kind::generate) {
                job["source"] = request.source ? files::utf8(request.source->path.lexically_relative(dataset::directory)) : "";
                job["seed"]   = request.seed;
                job["width"]  = request.parameters.width;
                job["height"] = request.parameters.height;
            } else if (request.kind == Kind::classify) job["input"] = files::utf8(std::filesystem::absolute(request.input).lexically_normal());
            else if (request.kind == Kind::train) job["target"] = request.training.steps;
            else if (request.kind == Kind::fix) job["category"] = request.category;
            events.push_back({EventKind::task, id, {}, 0, 0, job});
            if (request.kind == Kind::infer) {
                immediate.push_back(std::move(request));
                ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].yield_requested}.store(1, ::cuda::memory_order_release);
            } else queue.push_back(std::move(request));
        }
        condition.notify_all();
        if (visuals.notify) visuals.notify();
        return id;
    }
    void Session::observe(std::vector<Request> requests) {
        std::string signature;
        for (const auto& request : requests) signature += request.descriptor.id + request.descriptor.sha + request.file->sha;
        {
            const std::lock_guard lock{mutex};
            if (signature == observed || closing) return;
            observed = std::move(signature);
            for (auto entry = immediate.begin(); entry != immediate.end();) {
                if (!entry->transient) {
                    ++entry;
                    continue;
                }
                jobs.erase(entry->id);
                entry = immediate.erase(entry);
            }
        }
        for (auto& request : requests) {
            request.transient = true;
            enqueue(std::move(request));
        }
    }
    void Session::cancel(const std::uint64_t id) {
        std::optional<Request> removed;
        {
            const std::lock_guard lock{mutex};
            if (active && active->id == id) {
                const auto& job = jobs.at(id);
                if ((job.at("state") != "running" && job.at("state") != "queued") || active->kind == Kind::fix || active->kind == Kind::undo) return;
                if (active->kind == Kind::classify && job.contains("progress") && job.at("progress").value("stage", "") == "moving") return;
                interrupted.store(true);
                ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.store(1, ::cuda::memory_order_release);
                return;
            }
            for (auto* pending : {&queue, &immediate}) {
                const auto found = std::ranges::find(*pending, id, &Request::id);
                if (found != pending->end()) {
                    removed = std::move(*found);
                    pending->erase(found);
                    break;
                }
            }
        }
        if (removed) emit(*removed, "stopped");
    }
    void Session::shutdown() {
        {
            const std::lock_guard lock{mutex};
            closing = true;
            interrupted.store(true);
            for (auto& [id, job] : jobs)
                if (job.at("state") == "queued") job["state"] = "stopped";
            for (const auto& request : queue) events.push_back({EventKind::task, request.id, {}, 0, 0, jobs.at(request.id)});
            for (const auto& request : immediate) events.push_back({EventKind::task, request.id, {}, 0, 0, jobs.at(request.id)});
            queue.clear();
            immediate.clear();
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.store(1, ::cuda::memory_order_release);
        }
        condition.notify_all();
    }
    void Session::emit(const Request& request, const std::string_view state, nlohmann::json fields) {
        {
            const std::lock_guard lock{mutex};
            auto& job        = jobs[request.id];
            job["id"]        = request.id;
            job["kind"]      = static_cast<int>(request.kind);
            job["concept"]   = request.concept_key;
            job["image_sha"] = request.file ? request.file->sha : request.sha;
            job["model_sha"] = request.descriptor.sha;
            job["state"]     = state;
            for (auto& [key, value] : fields.items()) job[key] = std::move(value);
            events.push_back({EventKind::task, request.id, {}, 0, 0, job});
            if (request.transient && (state == "complete" || state == "failed" || state == "stopped")) jobs.erase(request.id);
        }
        if (visuals.notify) visuals.notify();
    }
    void Session::run() {
        std::unique_ptr<dataset::Lock> lease;
        std::string gpu_name{"gpu-"};
        const auto gpu_locks = std::filesystem::temp_directory_path() / "genesia-gpu";
        try {
            cudaDeviceProp properties;
            neural::check(cudaGetDeviceProperties(&properties, 0));
            for (const auto byte : properties.uuid.bytes) gpu_name += std::format("{:02x}", static_cast<unsigned char>(byte));
            for (;;) {
                Request request;
                {
                    std::unique_lock lock{mutex};
                    while (!closing && immediate.empty() && queue.empty()) {
                        lock.unlock();
                        if (lease) {
                            const dataset::Lock demand{gpu_name + "-wanted", false, gpu_locks};
                            if (!demand.acquired) {
                                release_generation();
                                predictions.reset();
                                lease.reset();
                                for (;;) {
                                    const dataset::Lock waiting{gpu_name + "-wanted", false, gpu_locks};
                                    if (waiting.acquired) break;
                                    std::unique_lock wait_lock{mutex};
                                    if (closing) break;
                                    condition.wait_for(wait_lock, std::chrono::milliseconds{100});
                                }
                            }
                        }
                        lock.lock();
                        condition.wait_for(lock, std::chrono::milliseconds{100}, [this] { return closing || !immediate.empty() || !queue.empty(); });
                    }
                    if (closing) break;
                    if (!immediate.empty()) {
                        request = std::move(immediate.front());
                        immediate.pop_front();
                    } else {
                        request = std::move(queue.front());
                        queue.pop_front();
                    }
                    active  = request;
                    started = std::chrono::steady_clock::now();
                    interrupted.store(false);
                    ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.store(0);
                }
                emit(request, "running");
                try {
                    bool cached{};
                    if (request.kind == Kind::infer) {
                        if (request.descriptor.id.empty()) request.descriptor = classifier::model(request.concept_key);
                        if (!request.file) {
                            dataset::Index index;
                            request.file = index.identify(request.input);
                        }
                        if (!predictions) predictions = std::make_unique<classifier::Predictions>();
                        cached = !request.refresh && predictions->find(request.descriptor, request.file->sha).has_value();
                    }
                    std::unique_ptr<dataset::Lock> demand;
                    while (!lease && !cached) {
                        auto attempt = std::make_unique<dataset::Lock>(gpu_name, false, gpu_locks);
                        if (attempt->acquired) {
                            lease = std::move(attempt);
                            break;
                        }
                        if (interrupted.load()) throw std::runtime_error{"Stopped while waiting for GPU"};
                        if (!demand) {
                            auto wanted = std::make_unique<dataset::Lock>(gpu_name + "-wanted", false, gpu_locks);
                            if (wanted->acquired) demand = std::move(wanted);
                        }
                        std::unique_lock lock{mutex};
                        condition.wait_for(lock, std::chrono::milliseconds{100});
                    }
                    demand.reset();
                    execute(request);
                } catch (const std::exception& failure) {
                    {
                        const std::lock_guard lock{mutex};
                        preview_sampling = false;
                    }
                    condition.notify_all();
                    emit(request, interrupted.load() ? "stopped" : "failed", {{"error", failure.what()}});
                }
                {
                    const std::lock_guard lock{mutex};
                    active.reset();
                }
                const dataset::Lock demand{gpu_name + "-wanted", false, gpu_locks};
                if (!demand.acquired) {
                    release_generation();
                    predictions.reset();
                    lease.reset();
                    for (;;) {
                        const dataset::Lock waiting{gpu_name + "-wanted", false, gpu_locks};
                        if (waiting.acquired) break;
                        std::unique_lock wait_lock{mutex};
                        if (closing) break;
                        condition.wait_for(wait_lock, std::chrono::milliseconds{100});
                    }
                }
                if (visuals.notify) visuals.notify();
            }
            release_generation();
            predictions.reset();
            lease.reset();
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
    void Session::execute(const Request& request) {
        if (request.kind == Kind::generate) {
            if (!generate(request)) emit(request, "stopped");
            return;
        }
        if (!predictions) predictions = std::make_unique<classifier::Predictions>();
        const auto progress    = [&](const nlohmann::json& value) { emit(request, "running", {{"progress", value}}); };
        const auto cooperative = [this] { yield(); };
        nlohmann::json result;
        switch (request.kind) {
        case Kind::train:
            release_generation();
            result = classifier::train(request.training, interrupted, progress, cooperative);
            break;
        case Kind::infer:
            {
                const auto descriptor = request.descriptor.id.empty() ? classifier::model(request.concept_key) : request.descriptor;
                if (classifier::model(descriptor.id).sha != descriptor.sha) throw std::runtime_error{"Classifier version changed"};
                dataset::Index index;
                const auto file       = request.file ? *request.file : index.identify(request.input);
                result                = predictions->infer(descriptor, file, request.refresh);
                auto identified       = request;
                identified.file       = file;
                identified.descriptor = descriptor;
                emit(identified, "complete", {{"result", std::move(result)}});
                return;
            }
        case Kind::audit: result = classifier::audit(request.concept_key, *predictions, request.refresh, interrupted, progress, cooperative); break;
        case Kind::fix: result = classifier::fix(request.concept_key, request.sha, request.category); break;
        case Kind::undo: result = classifier::undo(request.concept_key); break;
        case Kind::classify: result = classifier::classify(request.concept_key, request.input, *predictions, interrupted, progress, cooperative); break;
        case Kind::generate: break;
        }
        const bool committed = request.kind == Kind::fix || request.kind == Kind::undo || request.kind == Kind::classify;
        emit(request, interrupted.load() && !committed ? "stopped" : "complete", {{"result", std::move(result)}});
    }
    void Session::yield() {
        Request request;
        {
            const std::lock_guard lock{mutex};
            if (immediate.empty() || closing) return;
            request = std::move(immediate.front());
            immediate.pop_front();
        }
        emit(request, "running");
        try {
            execute(request);
        } catch (const std::exception& failure) {
            emit(request, "failed", {{"error", failure.what()}});
        }
        const std::lock_guard lock{mutex};
        if (!immediate.empty()) ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].yield_requested}.store(1, ::cuda::memory_order_release);
    }
    bool Session::generate(const Request& request) {
        if (request.source) {
            dataset::Index index;
            const auto source = index.identify(request.source->path);
            if (source.sha != request.source->sha || source.width != request.parameters.width || source.height != request.parameters.height) throw std::runtime_error{"Repaint source changed after submission"};
        }
        if (!model) {
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::loading));
            model = std::make_unique<sdxl::Model>(stream, defaults::checkpoint, defaults::cache);
            const std::lock_guard lock{mutex};
            model_ready = true;
        }
        const auto source_id = request.source ? std::optional{request.source->sha} : std::nullopt;
        if (!inference || inference->parameters != request.parameters || source_id != prepared_image) {
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this] { return !saving[0] && !saving[1]; });
            }
            inference.reset();
            if (request.source) {
                if (source_id != encoded_image) {
                    const auto pixels = read_image(request.source->path);
                    source_image      = std::make_unique<sdxl::ImageInput>(stream, pixels.pixels, pixels.width, pixels.height);
                    encoded_image     = source_id;
                }
                if (request.parameters.denoise > 0 && source_image->latent.empty()) {
                    const auto started = std::chrono::steady_clock::now();
                    model->encode(*source_image);
                    std::println(std::cerr, "ENCODE {:.3f}s", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
                }
            }
            prepared_image = source_id;
            if (visuals.publish && (!snapshots || snapshots->width != request.parameters.width || snapshots->height != request.parameters.height)) {
                auto next = std::make_shared<sdxl::Snapshots>(stream, request.parameters.width, request.parameters.height);
                stream.sync();
                const std::lock_guard lock{mutex};
                snapshots     = std::move(next);
                preview_ready = false;
            }
            inference = std::make_unique<sdxl::Inference>(*model, request.parameters, control.data()[0], snapshots.get(), request.source ? source_image.get() : nullptr);
            std::println(std::cerr, "READY prepare={:.3f}s cache={}/{} memory={:.2f}GiB", inference->prepare_seconds, inference->cache_hits, inference->cache_misses, inference->resident_bytes / double(1ull << 30));
            std::cerr.flush();
            if (visuals.prepare) visuals.prepare(false, request.parameters.width, request.parameters.height, stream);
            iteration = 0;
        }
        if (!::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.load()) {
            const auto slot = iteration++ % 2;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this, slot] { return !saving[slot]; });
            }
            {
                std::unique_lock lock{mutex};
                if (visuals.publish && !preview_ready && request.parameters.denoise > 0) {
                    preview_prepare = true;
                    condition.notify_all();
                    condition.wait(lock, [this] { return preview_ready || !error.empty(); });
                    if (!error.empty()) throw std::runtime_error{error};
                }
                preview_sampling = visuals.publish && request.parameters.denoise > 0;
            }
            condition.notify_all();
            const auto& output = inference->generate(request.seed, [this] { yield(); });
            {
                const std::lock_guard lock{mutex};
                preview_sampling = false;
            }
            condition.notify_all();
            if (output.cancelled) interrupted.store(true);
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this] { return !preview_working; });
                if (!error.empty()) throw std::runtime_error{error};
                // Reuse snapshots only after both GPU streams have finished with them.
                if (snapshots)
                    for (auto& snapshot : snapshots->slots) ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{snapshot.state}.store(std::uint32_t(sdxl::SnapshotState::free), ::cuda::memory_order_release);
            }
            if (!output.cancelled) {
                Record record{request.parameters, request.seed, {}, std::filesystem::path{defaults::checkpoint}.filename(), request.prompt, request.catalog, request.source ? request.source->path.lexically_relative(dataset::directory) : std::filesystem::path{}};
                const auto ready = visuals.publish ? visuals.publish(false, output.device_pixels, output.width, output.height, output.stream, slot) : 0;
                emit(request, "saving");
                {
                    const std::lock_guard lock{mutex};
                    if (ready) events.push_back({EventKind::generated, request.id, record, slot, ready});
                    saving[slot] = true;
                    files.push_back({request, &output, record, slot});
                }
                if (visuals.notify) visuals.notify();
                condition.notify_all();
                std::println(std::cerr, "GENERATE seed={} sample={:.3f}s decode={:.3f}s", request.seed, output.sample_seconds, output.decode_seconds);
                std::cerr.flush();
            }
            return !output.cancelled;
        }
        return false;
    }
    void Session::release_generation() {
        {
            std::unique_lock lock{mutex};
            condition.wait(lock, [this] { return !saving[0] && !saving[1] && !preview_working; });
            preview_sampling = false;
            if (visuals.publish && snapshots && !preview_done) {
                preview_release = true;
                condition.notify_all();
                condition.wait(lock, [this] { return !preview_release || preview_done; });
            }
            model_ready = false;
        }
        inference.reset();
        source_image.reset();
        snapshots.reset();
        model.reset();
        encoded_image.reset();
        prepared_image.reset();
        if (visuals.publish) preview_stream.sync();
        stream.sync();
        cudaMemPool_t pool;
        neural::check(cudaDeviceGetDefaultMemPool(&pool, 0));
        neural::check(cudaMemPoolTrimTo(pool, 0));
    }
    void Session::preview_images() {
        std::unique_ptr<sdxl::Preview> decoder;
        std::shared_ptr<sdxl::Snapshots> source;
        std::optional<PreviewFrame> in_flight;
        int reading        = -1;
        auto next_snapshot = std::chrono::steady_clock::now();
        std::uint64_t task = std::numeric_limits<std::uint64_t>::max();
        try {
            for (;;) {
                std::unique_lock lock{mutex};
                if (in_flight) {
                    const auto completion = cudaEventQuery(preview_finished.get());
                    if (completion == cudaSuccess) {
                        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{source->slots.data()[reading].state}.store(std::uint32_t(sdxl::SnapshotState::free), ::cuda::memory_order_release);
                        if (in_flight->ready) previews.push_back(*in_flight);
                        in_flight.reset();
                        preview_working = false;
                        condition.notify_all();
                        if (visuals.notify) visuals.notify();
                    } else if (completion != cudaErrorNotReady) neural::check(completion);
                }
                if (preview_closing && !in_flight) break;
                if (preview_release && !in_flight) {
                    lock.unlock();
                    decoder.reset();
                    source.reset();
                    lock.lock();
                    preview_release = false;
                    preview_ready   = false;
                    condition.notify_all();
                }
                if (preview_prepare && !in_flight) {
                    preview_working  = true;
                    source           = snapshots;
                    const int width  = source->width;
                    const int height = source->height;
                    lock.unlock();
                    if (!decoder || decoder->width != width || decoder->height != height) {
                        decoder.reset();
                        decoder = std::make_unique<sdxl::Preview>(preview_stream, model->vae, defaults::cache, width, height);
                    }
                    visuals.prepare(true, width, height, preview_stream);
                    lock.lock();
                    preview_prepare = false;
                    preview_ready   = true;
                    preview_working = false;
                    condition.notify_all();
                    continue;
                }
                const auto stage    = static_cast<sdxl::Stage>(::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.load(::cuda::memory_order_acquire));
                const bool sampling = preview_sampling && active && preview_visible && preview_enabled && !closing && stage == sdxl::Stage::sampling && !::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.load(::cuda::memory_order_acquire);
                if (sampling) {
                    source = snapshots;
                    if (task != active->id) {
                        task          = active->id;
                        next_snapshot = std::chrono::steady_clock::now();
                    }
                    const auto now = std::chrono::steady_clock::now();
                    int free = -1, newest = -1;
                    bool requested{};
                    for (int i = 0; i < 3; ++i) {
                        const auto state = sdxl::SnapshotState(::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{source->slots.data()[i].state}.load(::cuda::memory_order_acquire));
                        if (state == sdxl::SnapshotState::free) free = i;
                        if (state == sdxl::SnapshotState::requested) requested = true;
                        if (state == sdxl::SnapshotState::ready && (newest < 0 || source->slots.data()[i].step > source->slots.data()[newest].step)) newest = i;
                    }
                    if (now >= next_snapshot && free >= 0 && !requested) {
                        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{source->slots.data()[free].state}.store(std::uint32_t(sdxl::SnapshotState::requested), ::cuda::memory_order_release);
                        next_snapshot = now + std::chrono::milliseconds{defaults::preview_interval_ms};
                    }
                    if (newest >= 0) {
                        for (int i = 0; i < 3; ++i) {
                            auto state = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{source->slots.data()[i].state};
                            if (i != newest && state.load(::cuda::memory_order_acquire) == std::uint32_t(sdxl::SnapshotState::ready)) state.store(std::uint32_t(sdxl::SnapshotState::free), ::cuda::memory_order_release);
                        }
                        if (!in_flight) {
                            std::size_t slot = 0;
                            for (; slot < 2; ++slot) {
                                if (visuals.available(slot)) break;
                            }
                            if (slot < 2) {
                                reading         = newest;
                                const auto step = source->slots.data()[reading].step;
                                ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{source->slots.data()[reading].state}.store(std::uint32_t(sdxl::SnapshotState::reading), ::cuda::memory_order_release);
                                preview_working       = true;
                                const bool from_image = active->source.has_value();
                                lock.unlock();
                                decoder->decode(source->latent.data() + std::size_t(reading) * (source->width / 8) * (source->height / 8) * 4);
                                const auto ready = visuals.publish(true, decoder->pixels.data(), decoder->width, decoder->height, preview_stream, slot);
                                neural::check(cudaEventRecord(preview_finished.get(), preview_stream.get()));
                                in_flight = PreviewFrame{task, step, decoder->width, decoder->height, slot, ready, from_image};
                                lock.lock();
                            }
                        }
                    }
                } else if (source) {
                    for (auto& slot : source->slots) {
                        auto state = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{slot.state};
                        if (state.load(::cuda::memory_order_acquire) == std::uint32_t(sdxl::SnapshotState::ready)) state.store(std::uint32_t(sdxl::SnapshotState::free), ::cuda::memory_order_release);
                    }
                }
                if (preview_sampling || in_flight) condition.wait_for(lock, std::chrono::milliseconds{4});
                else condition.wait(lock, [this] { return preview_closing || preview_prepare || preview_sampling || preview_release; });
            }
        } catch (const std::exception& failure) {
            preview_stream.sync();
            {
                const std::lock_guard lock{mutex};
                error           = failure.what();
                preview_working = false;
            }
            shutdown();
            if (visuals.notify) visuals.notify();
        }
        {
            const std::lock_guard lock{mutex};
            preview_done = true;
        }
        condition.notify_all();
    }

    void Session::write_files() {
        for (;;) {
            FileTask task;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this] { return io_closing || !files.empty(); });
                if (files.empty()) break;
                task = std::move(files.front());
                files.pop_front();
            }
            std::vector<std::string> selected;
            bool saved{};
            try {
                task.record.path = std::filesystem::absolute(save_image(*task.output, task.record));
                {
                    const std::lock_guard lock{mutex};
                    events.push_back({EventKind::saved, task.request.id, task.record});
                    if (!closing) selected = activated;
                }
                saved = true;
            } catch (const std::exception& failure) {
                emit(task.request, "failed", {{"error", failure.what()}});
            }
            if (saved) {
                for (const auto& key : selected) {
                    Request inference_request;
                    inference_request.kind        = Kind::infer;
                    inference_request.concept_key = key;
                    inference_request.input       = task.record.path;
                    try {
                        enqueue(std::move(inference_request));
                    } catch (const std::exception& failure) {
                        std::println(std::cerr, "Post-save inference: {}", failure.what());
                    }
                }
                emit(task.request, "complete", {{"result", {{"path", files::utf8(task.record.path)}, {"seed", task.record.seed}}}});
            }
            {
                const std::lock_guard lock{mutex};
                saving[task.slot] = false;
            }
            condition.notify_all();
            if (visuals.notify) visuals.notify();
        }
    }
} // namespace genesia::work
