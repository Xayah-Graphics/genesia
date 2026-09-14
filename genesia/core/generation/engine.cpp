module;
#include "../models/sdxl/control.h"
#include <genesia/cuda.h>
module genesia.generation.engine;
import genesia.compute.device;
import genesia.project;
import std;
namespace genesia::generation {
    Engine::Engine(Visuals hooks, std::function<void(runtime::Event)> events, std::function<void(runtime::PreviewFrame)> frames) : visuals{std::move(hooks)}, report{std::move(events)}, preview{std::move(frames)}, stream{::cuda::devices[0]}, control{stream, ::cuda::pinned_default_memory_pool(), 1, ::cuda::no_init} {
        std::construct_at(control.data());
        if (visuals.publish) {
            int least, greatest;
            compute::check(cudaDeviceGetStreamPriorityRange(&least, &greatest));
            preview_stream = ::cuda::stream{::cuda::devices[0], least};
            compute::check(cudaEventCreateWithFlags(std::out_ptr(preview_finished), cudaEventDisableTiming));
        }
        if (visuals.publish) preview_worker = std::jthread{[this] { preview_images(); }};
    }
    Engine::~Engine() {
        finish();
    }
    std::optional<SavedImage> Engine::generate(dataset::Index& index, const std::uint64_t id, const runtime::Generate& request, const std::atomic_bool& interrupted) {
        {
            const std::lock_guard lock{mutex};
            active  = Active{id, request};
            started = std::chrono::steady_clock::now();
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.store(interrupted.load() ? 1u : 0u);
        }

        if (!model) {
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::loading));
            model = std::make_unique<sdxl::Model>(stream, project::checkpoint, project::cache);
            const std::lock_guard lock{mutex};
            model_ready = true;
        }
        const auto source_id = request.source ? std::optional{request.source->sha} : std::nullopt;
        if (!inference || inference->parameters != request.parameters || source_id != prepared_image) {
            inference.reset();
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::preparing));
            try {
                model->apply_loras(request.parameters.loras);
            } catch (...) {
                release();
                throw;
            }
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
        if (interrupted.load()) cancel();
        if (!::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.load()) {
            const auto slot = iteration++ % 2;
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
            const auto& output = [&]() -> const sdxl::Output& {
                try {
                    return inference->generate(request.seed);
                } catch (...) {
                    {
                        const std::lock_guard lock{mutex};
                        preview_sampling = false;
                    }
                    condition.notify_all();
                    throw;
                }
            }();
            {
                const std::lock_guard lock{mutex};
                preview_sampling = false;
            }
            condition.notify_all();
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this] { return !preview_working; });
                if (!error.empty()) throw std::runtime_error{error};
                // Reuse snapshots only after both GPU streams have finished with them.
                if (snapshots)
                    for (auto& snapshot : snapshots->slots) ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{snapshot.state}.store(std::uint32_t(sdxl::SnapshotState::free), ::cuda::memory_order_release);
            }
            if (!output.cancelled) {
                Record record{request.parameters, request.seed, {}, std::filesystem::path{project::checkpoint}.filename(), prompt::store(request.prompt, *request.catalog), request.source ? request.source->path.lexically_relative(project::directory) : std::filesystem::path{}};
                const auto ready = visuals.publish ? visuals.publish(false, output.device_pixels, output.width, output.height, output.stream, slot) : nullptr;
                report({runtime::EventKind::task, id, {}, {}, {.id = id, .state = runtime::State::saving}});
                if (ready) report({runtime::EventKind::generated, id, record, ready});
                auto file   = save_image(index, output, record);
                record.path = file.path;
                report({.kind = runtime::EventKind::saved, .id = id, .record = record, .file = file});
                std::println(std::cerr, "GENERATE seed={} sample={:.3f}s decode={:.3f}s", request.seed, output.sample_seconds, output.decode_seconds);
                return SavedImage{std::move(record), std::move(file), {output.pixels.data(), output.pixels.size()}};
            }
            return {};
        }
        return {};
    }
    void Engine::cancel() {
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.store(1, ::cuda::memory_order_release);
    }
    void Engine::configure(const bool enabled, const bool visible) {
        const std::lock_guard lock{mutex};
        preview_enabled = enabled;
        preview_visible = visible;
    }
    runtime::GenerationProgress Engine::observe() {
        const std::lock_guard lock{mutex};
        if (!error.empty()) throw std::runtime_error{error};
        return {active ? active->id : 0, static_cast<runtime::GenerationStage>(::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.load()), ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].completed}.load(), active ? active->request.parameters.steps : 0, model_ready, started};
    }
    void Engine::finish() {
        if (std::exchange(finished, true)) return;
        cancel();
        release();
        {
            const std::lock_guard lock{mutex};
            preview_closing = true;
        }
        condition.notify_all();
        if (preview_worker.joinable()) preview_worker.join();
    }
    void Engine::release() {
        {
            std::unique_lock lock{mutex};
            condition.wait(lock, [this] { return !preview_working; });
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
        compute::check(cudaDeviceGetDefaultMemPool(&pool, 0));
        compute::check(cudaMemPoolTrimTo(pool, 0));
    }
    void Engine::preview_images() {
        std::unique_ptr<sdxl::Preview> decoder;
        std::shared_ptr<sdxl::Snapshots> source;
        std::optional<runtime::PreviewFrame> in_flight;
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
                        if (in_flight->frame) preview(*in_flight);
                        in_flight.reset();
                        preview_working = false;
                        condition.notify_all();
                        if (visuals.notify) visuals.notify();
                    } else if (completion != cudaErrorNotReady) compute::check(completion);
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
                        decoder = std::make_unique<sdxl::Preview>(preview_stream, model->vae, project::cache, width, height);
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
                const bool sampling = preview_sampling && active && preview_visible && preview_enabled && !preview_closing && stage == sdxl::Stage::sampling && !::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.load(::cuda::memory_order_acquire);
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
                                const bool from_image = active->request.source.has_value();
                                lock.unlock();
                                decoder->decode(source->latent.data() + std::size_t(reading) * (source->width / 8) * (source->height / 8) * 4);
                                const auto ready = visuals.publish(true, decoder->pixels.data(), decoder->width, decoder->height, preview_stream, slot);
                                compute::check(cudaEventRecord(preview_finished.get(), preview_stream.get()));
                                in_flight = runtime::PreviewFrame{task, step, decoder->width, decoder->height, ready, from_image};
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
            cancel();
            if (visuals.notify) visuals.notify();
        }
        {
            const std::lock_guard lock{mutex};
            preview_done = true;
        }
        condition.notify_all();
    }

} // namespace genesia::generation
