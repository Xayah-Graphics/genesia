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
        save_worker = std::jthread{[this] { save_images(); }};
    }
    Engine::~Engine() {
        finish();
    }
    bool Engine::generate(const std::uint64_t id, const runtime::Generate& request, const std::atomic_bool& interrupted) {
        {
            const std::lock_guard lock{mutex};
            if (save_error) std::rethrow_exception(save_error);
            active = Active{id, request.parameters.steps};
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.store(interrupted.load() ? 1u : 0u);
        }

        if (!model) {
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::loading));
            model = std::make_unique<sdxl::Model>(stream, project::checkpoint, project::cache);
            const std::lock_guard lock{mutex};
            model_ready = true;
        }
        if (!inference || inference->parameters != request.parameters) {
            const auto started = std::chrono::steady_clock::now();
            flush();
            inference.reset();
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::preparing));
            if (visuals.publish && (!snapshots || snapshots->width != request.parameters.width || snapshots->height != request.parameters.height)) {
                auto next = std::make_shared<sdxl::Snapshots>(stream, request.parameters.width, request.parameters.height);
                stream.sync();
                const std::lock_guard lock{mutex};
                snapshots = std::move(next);
            }
            try {
                if (!decoder || decoder->width != request.parameters.width || decoder->height != request.parameters.height) {
                    decoder.reset();
                    decoder = std::make_unique<sdxl::Decoder>(visuals.publish ? preview_stream : stream, model->vae, project::cache, request.parameters.width, request.parameters.height);
                }
                inference     = std::make_unique<sdxl::Inference>(*model, request.parameters, control.data()[0], snapshots.get());
                saving_output = std::make_unique<sdxl::Output>(stream, request.parameters.width, request.parameters.height);
            } catch (...) {
                release();
                throw;
            }
            if (visuals.prepare) {
                visuals.prepare(false);
                visuals.prepare(true);
            }
            stream.sync();
            decoder->stream.sync();
            cudaMemPool_t pool;
            compute::check(cudaDeviceGetDefaultMemPool(&pool, 0));
            compute::check(cudaMemPoolTrimTo(pool, 0));
            std::size_t used, reserved, free, total;
            compute::check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
            compute::check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
            compute::check(cudaMemGetInfo(&free, &total));
            const auto prepared = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::println(std::cerr, "READY prepare={:.3f}s cache={}/{} pool_used={:.2f}GiB pool_reserved={:.2f}GiB device_used={:.2f}GiB", prepared, inference->cache_hits + decoder->cache_hits, inference->cache_misses + decoder->cache_misses, used / double(1ull << 30), reserved / double(1ull << 30), (total - free) / double(1ull << 30));
            std::cerr.flush();
            iteration = 0;
        }
        if (interrupted.load()) cancel();
        if (!::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.load()) {
            const auto slot = iteration++ % 2;
            {
                std::unique_lock lock{mutex};
                preview_sampling = static_cast<bool>(visuals.publish);
            }
            condition.notify_all();
            auto& output = [&]() -> sdxl::Output& {
                try {
                    return inference->sample(request.seed);
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
            if (::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.load()) output.cancelled = true;
            if (!output.cancelled) {
                // The last preview has completed; final decoding owns the same buffers.
                decoder->decode(inference->latent.data());
                compute::check(cudaEventSynchronize(decoder->finished.get()));
                if (::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.load()) {
                    ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::cancelled));
                    return {};
                }
                float milliseconds;
                compute::check(cudaEventElapsedTime(&milliseconds, decoder->started.get(), decoder->finished.get()));
                output.decode_seconds = milliseconds * 0.001;
                output.device_pixels  = decoder->pixels.data();
                output.stream         = decoder->stream;
                ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::transferring));
                ::cuda::copy_bytes(decoder->stream, decoder->pixels, output.pixels);
                decoder->stream.sync();
                ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::complete));
                Record record{request.parameters, request.seed, {}, std::filesystem::path{project::checkpoint}.filename()};
                const auto ready = visuals.publish ? visuals.publish(id, false, output.device_pixels, output.width, output.height, output.stream, slot) : nullptr;
                report({runtime::EventKind::task, id, {}, {}, {.id = id, .state = runtime::State::saving}});
                if (ready) report({runtime::EventKind::generated, id, record, ready});
                {
                    std::unique_lock lock{mutex};
                    condition.wait(lock, [this] { return !save_pending; });
                    if (save_error) std::rethrow_exception(save_error);
                    // The writer owns these pixels until it has reported the saved image.
                    std::swap(output.pixels, saving_output->pixels);
                    saving_output->sample_seconds = output.sample_seconds;
                    saving_output->decode_seconds = output.decode_seconds;
                    saving_record                 = std::move(record);
                    saving_id                     = id;
                    save_pending                  = true;
                }
                condition.notify_all();
                return true;
            }
            return {};
        }
        return {};
    }
    void Engine::flush() {
        std::unique_lock lock{mutex};
        condition.wait(lock, [this] { return !save_pending; });
        if (save_error) std::rethrow_exception(std::exchange(save_error, {}));
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
        return {static_cast<runtime::GenerationStage>(::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.load()), ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].completed}.load(), active ? active->steps : 0, model_ready};
    }
    void Engine::finish() {
        if (std::exchange(finished, true)) return;
        cancel();
        {
            std::unique_lock lock{mutex};
            condition.wait(lock, [this] { return !save_pending; });
            save_closing = true;
        }
        condition.notify_all();
        if (save_worker.joinable()) save_worker.join();
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
            model_ready      = false;
        }
        inference.reset();
        snapshots.reset();
        decoder.reset();
        model.reset();
        if (visuals.publish) preview_stream.sync();
        stream.sync();
        cudaMemPool_t pool;
        compute::check(cudaDeviceGetDefaultMemPool(&pool, 0));
        compute::check(cudaMemPoolTrimTo(pool, 0));
    }
    void Engine::preview_images() {
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
                                preview_working = true;
                                lock.unlock();
                                decoder->decode(source->latent.data() + std::size_t(reading) * (source->width / 8) * (source->height / 8) * 4);
                                const auto ready = visuals.publish(task, true, decoder->pixels.data(), decoder->width, decoder->height, preview_stream, slot);
                                compute::check(cudaEventRecord(preview_finished.get(), preview_stream.get()));
                                in_flight = runtime::PreviewFrame{task, step, decoder->width, decoder->height, ready};
                                lock.lock();
                            }
                        }
                    }
                } else if (source) {
                    for (auto& slot : source->slots) {
                        auto state = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{slot.state};
                        if (state.load(::cuda::memory_order_acquire) == std::uint32_t(sdxl::SnapshotState::ready)) state.store(std::uint32_t(sdxl::SnapshotState::free), ::cuda::memory_order_release);
                    }
                    if (!in_flight) source.reset();
                }
                if (preview_sampling || in_flight) condition.wait_for(lock, std::chrono::milliseconds{4});
                else condition.wait(lock, [this] { return preview_closing || preview_sampling; });
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
        condition.notify_all();
    }

    void Engine::save_images() {
        std::unique_lock lock{mutex};
        for (;;) {
            condition.wait(lock, [this] { return save_pending || save_closing; });
            if (save_closing) return;
            lock.unlock();
            try {
                const auto started = std::chrono::steady_clock::now();
                saving_record.path = save_image(*saving_output, saving_record);
                const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                report({.kind = runtime::EventKind::saved, .id = saving_id, .record = saving_record, .timing = {saving_output->sample_seconds, saving_output->decode_seconds, seconds}});
            } catch (...) {
                const std::lock_guard failure_lock{mutex};
                save_error = std::current_exception();
                cancel();
            }
            lock.lock();
            save_pending = false;
            condition.notify_all();
        }
    }
} // namespace genesia::generation
