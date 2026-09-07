module;
#include <GLFW/glfw3.h>

#include "../../core/sdxl/control.h"
#include <genesia/cuda.h>
module genesia.editor.session;
import genesia.generation.configuration;
import genesia.generation.output;
import genesia.sdxl;
import genesia.sdxl.preview;
import genesia.neural.inference_runtime;
import genesia.editor.platform.interop;
import std;

namespace genesia::editor {
    ::cuda::stream priority_stream(const bool high) {
        int least, greatest;
        neural::check(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        return ::cuda::stream{::cuda::devices[0], high ? greatest : least};
    }

    Session::Session(Configuration configuration, Interop& bridge, Interop& preview_bridge) : configuration{std::move(configuration)}, interop{bridge}, preview_interop{preview_bridge}, stream{priority_stream(true)}, control{stream, ::cuda::pinned_default_memory_pool(), 1, ::cuda::no_init}, preview_enabled{this->configuration.preview.enabled} {
        std::construct_at(control.data());
        preview_stream = priority_stream(false);
        neural::check(cudaEventCreateWithFlags(&preview_finished, cudaEventDisableTiming));
        preview_worker = std::jthread{[this] { preview_images(); }};
        worker         = std::jthread{[this] { generate(); }};
        io             = std::jthread{[this] { write_files(); }};
    }

    Session::~Session() {
        shutdown();
        worker.join();
        {
            const std::lock_guard lock{mutex};
            io_closing      = true;
            preview_closing = true;
        }
        condition.notify_all();
        io.join();
        preview_worker.join();
        model.reset();
        preview_stream.sync();
        cudaEventDestroy(preview_finished);
        stream.sync();
    }

    void Session::enqueue(sdxl::Parameters parameters, const std::uint64_t seed) {
        {
            const std::lock_guard lock{mutex};
            queue.push_back({next_id++, std::move(parameters), seed});
        }
        condition.notify_all();
    }

    void Session::stop() {
        const std::lock_guard lock{mutex};
        paused = true;
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.store(1, ::cuda::memory_order_release);
    }

    void Session::resume() {
        {
            const std::lock_guard lock{mutex};
            paused = false;
        }
        condition.notify_all();
    }

    void Session::load(const std::uint64_t id, std::filesystem::path path) {
        {
            const std::lock_guard lock{mutex};
            files.push_back({id, nullptr, {}, std::move(path)});
        }
        condition.notify_all();
    }

    void Session::shutdown() {
        {
            const std::lock_guard lock{mutex};
            closing = true;
            queue.clear();
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.store(1, ::cuda::memory_order_release);
        }
        condition.notify_all();
    }

    void Session::generate() {
        try {
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::loading));
            const auto load_started = std::chrono::steady_clock::now();
            model                   = std::make_unique<sdxl::Model>(stream, configuration.checkpoint, configuration.cache);
            const auto load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_started).count();
            std::unique_ptr<sdxl::Inference> inference;
            struct PendingWrites final {
                Session& session;
                ~PendingWrites() {
                    std::unique_lock lock{session.mutex};
                    session.condition.wait(lock, [this] { return !session.saving[0] && !session.saving[1]; });
                }
            } pending_writes{*this};
            std::size_t iteration{};
            const auto directory = session_directory(configuration.output);
            {
                const std::lock_guard lock{mutex};
                model_ready = true;
            }
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::idle));
            glfwPostEmptyEvent();
            for (;;) {
                Request request;
                {
                    std::unique_lock lock{mutex};
                    condition.wait(lock, [this] { return closing || (!paused && !queue.empty()); });
                    if (closing) break;
                    request = std::move(queue.front());
                    queue.pop_front();
                    active  = request;
                    started = std::chrono::steady_clock::now();
                    ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.store(0);
                }
                glfwPostEmptyEvent();
                if (!inference || inference->parameters != request.parameters) {
                    {
                        std::unique_lock lock{mutex};
                        condition.wait(lock, [this] { return !saving[0] && !saving[1]; });
                    }
                    inference.reset();
                    if (!snapshots || snapshots->width != request.parameters.width || snapshots->height != request.parameters.height) {
                        auto next = std::make_shared<sdxl::Snapshots>(stream, request.parameters.width, request.parameters.height);
                        stream.sync();
                        const std::lock_guard lock{mutex};
                        snapshots     = std::move(next);
                        preview_ready = false;
                    }
                    ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::preparing));
                    inference = std::make_unique<sdxl::Inference>(*model, request.parameters, control.data()[0], snapshots.get());
                    interop.prepare(request.parameters.width, request.parameters.height, stream);
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
                        if (!preview_ready) {
                            preview_prepare = true;
                            condition.notify_all();
                            condition.wait(lock, [this] { return preview_ready || !error.empty(); });
                            if (!error.empty()) throw std::runtime_error{error};
                        }
                        preview_sampling = true;
                    }
                    std::size_t free_bytes, total_bytes;
                    neural::check(cudaMemGetInfo(&free_bytes, &total_bytes));
                    inference->resident_bytes = total_bytes - free_bytes;
                    condition.notify_all();
                    const auto generation_started = std::chrono::steady_clock::now();
                    const auto& output            = inference->generate(request.seed);
                    {
                        const std::lock_guard lock{mutex};
                        preview_sampling = false;
                    }
                    condition.notify_all();
                    if (!output.cancelled) {
                        const auto generated = std::chrono::steady_clock::now();
                        const auto ready     = interop.publish(output.device_pixels, output.width, output.height, output.stream, slot);
                        Record record{request.parameters, request.seed, directory / std::format("{:03}-{}", request.id, request.seed), load_seconds, inference->prepare_seconds, output.sample_seconds, output.decode_seconds, inference->resident_bytes, inference->cache_hits, inference->cache_misses, generation_started};
                        auto preview = thumbnail(output);
                        {
                            const std::lock_guard lock{mutex};
                            events.push_back({EventKind::generated, request.id, record, std::move(preview), slot, ready, generated});
                            saving[slot] = true;
                            files.push_back({request.id, &output, record, {}, slot});
                        }
                        glfwPostEmptyEvent();
                        condition.notify_all();
                        std::println("GENERATE seed={} sample={:.3f}s decode={:.3f}s", request.seed, output.sample_seconds, output.decode_seconds);
                        std::cout.flush();
                    }
                    {
                        std::unique_lock lock{mutex};
                        condition.wait(lock, [this] { return !preview_working; });
                        if (!error.empty()) throw std::runtime_error{error};
                        // Reuse snapshots only after both GPU streams have finished with them.
                        for (auto& snapshot : snapshots->slots) ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{snapshot.state}.store(std::uint32_t(sdxl::SnapshotState::free), ::cuda::memory_order_release);
                    }
                }
                {
                    const std::lock_guard lock{mutex};
                    active.reset();
                }
                glfwPostEmptyEvent();
            }
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this] { return !saving[0] && !saving[1]; });
            }
            stream.sync();
        } catch (const std::exception& failure) {
            stream.sync();
            std::unique_lock lock{mutex};
            preview_sampling = false;
            condition.notify_all();
            condition.wait(lock, [this] { return !preview_working; });
            error = failure.what();
            active.reset();
            paused = true;
        }
        {
            const std::lock_guard lock{mutex};
            worker_done = true;
        }
        glfwPostEmptyEvent();
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
                    const auto completion = cudaEventQuery(preview_finished);
                    if (completion == cudaSuccess) {
                        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{source->slots.data()[reading].state}.store(std::uint32_t(sdxl::SnapshotState::free), ::cuda::memory_order_release);
                        previews.push_back(*in_flight);
                        in_flight.reset();
                        preview_working = false;
                        condition.notify_all();
                        glfwPostEmptyEvent();
                    } else if (completion != cudaErrorNotReady) neural::check(completion);
                }
                if (preview_closing && !in_flight) break;
                if (preview_prepare && !in_flight) {
                    preview_working  = true;
                    source           = snapshots;
                    const int width  = source->width;
                    const int height = source->height;
                    lock.unlock();
                    if (!decoder || decoder->width != width || decoder->height != height) {
                        decoder.reset();
                        decoder = std::make_unique<sdxl::Preview>(preview_stream, model->vae, configuration.cache, width, height);
                    }
                    preview_interop.prepare(width, height, preview_stream);
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
                        next_snapshot = now + std::chrono::milliseconds{configuration.preview.interval_ms};
                    }
                    if (newest >= 0) {
                        for (int i = 0; i < 3; ++i) {
                            auto state = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{source->slots.data()[i].state};
                            if (i != newest && state.load(::cuda::memory_order_acquire) == std::uint32_t(sdxl::SnapshotState::ready)) state.store(std::uint32_t(sdxl::SnapshotState::free), ::cuda::memory_order_release);
                        }
                        if (!in_flight) {
                            std::size_t slot = 0;
                            for (; slot < preview_interop.slots.size(); ++slot) {
                                const auto& target = preview_interop.slots[slot];
                                if (!target.value || target.timeline.getCounterValue() >= target.value + 1) break;
                            }
                            if (slot < preview_interop.slots.size()) {
                                reading         = newest;
                                const auto step = source->slots.data()[reading].step;
                                ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{source->slots.data()[reading].state}.store(std::uint32_t(sdxl::SnapshotState::reading), ::cuda::memory_order_release);
                                preview_working = true;
                                lock.unlock();
                                decoder->decode(source->latent.data() + std::size_t(reading) * (source->width / 8) * (source->height / 8) * 4);
                                const auto ready = preview_interop.publish(decoder->pixels.data(), decoder->width, decoder->height, preview_stream, slot);
                                neural::check(cudaEventRecord(preview_finished, preview_stream.get()));
                                in_flight = PreviewFrame{task, step, decoder->width, decoder->height, slot, ready};
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
                else condition.wait(lock, [this] { return preview_closing || preview_prepare || preview_sampling; });
            }
        } catch (const std::exception& failure) {
            preview_stream.sync();
            const std::lock_guard lock{mutex};
            error           = failure.what();
            paused          = true;
            preview_working = false;
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.store(1, ::cuda::memory_order_release);
            condition.notify_all();
            glfwPostEmptyEvent();
        }
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
            try {
                if (task.output) {
                    save(*task.output, task.record);
                    const std::lock_guard lock{mutex};
                    saving[task.slot] = false;
                    events.push_back({EventKind::saved, task.id});
                } else {
                    auto image = read_image(task.path);
                    const std::lock_guard lock{mutex};
                    events.push_back({EventKind::loaded, task.id, {}, std::move(image)});
                }
            } catch (const std::exception& failure) {
                const std::lock_guard lock{mutex};
                error  = failure.what();
                paused = true;
                if (task.output) saving[task.slot] = false;
            }
            condition.notify_all();
            glfwPostEmptyEvent();
        }
    }
} // namespace genesia::editor
