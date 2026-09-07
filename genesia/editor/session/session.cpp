module;
#include <genesia/cuda.h>
#include <GLFW/glfw3.h>
#include "../../core/sdxl/control.h"
module genesia.editor.session;
import genesia.generation.configuration;
import genesia.generation.output;
import genesia.sdxl;
import genesia.editor.platform.interop;
import std;

namespace genesia::editor {
    Session::Session(Configuration configuration, Interop& bridge)
        : configuration{std::move(configuration)}, interop{bridge}, stream{::cuda::devices[0]},
          control{stream, ::cuda::pinned_default_memory_pool(), 1, ::cuda::no_init} {
        std::construct_at(control.data());
        worker = std::jthread{[this] { generate(); }};
        io = std::jthread{[this] { write_files(); }};
    }

    Session::~Session() {
        shutdown();
        worker.join();
        {
            const std::lock_guard lock{mutex};
            io_closing = true;
        }
        condition.notify_all();
        io.join();
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
            sdxl::Model model{stream, configuration.checkpoint, configuration.cache};
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
                    active = request;
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
                    ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].stage}.store(static_cast<std::uint32_t>(sdxl::Stage::preparing));
                    inference = std::make_unique<sdxl::Inference>(model, request.parameters, control.data()[0]);
                    iteration = 0;
                }
                if (!::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.data()[0].cancel}.load()) {
                    const auto slot = iteration++ % 2;
                    {
                        std::unique_lock lock{mutex};
                        condition.wait(lock, [this, slot] { return !saving[slot]; });
                    }
                    const auto generation_started = std::chrono::steady_clock::now();
                    const auto& output = inference->generate(request.seed);
                    if (!output.cancelled) {
                        const auto generated = std::chrono::steady_clock::now();
                        const auto ready = interop.publish(output, slot);
                        Record record{request.parameters, request.seed, directory / std::format("{:03}-{}", request.id, request.seed), load_seconds, inference->prepare_seconds,
                            output.sample_seconds, output.decode_seconds, inference->resident_bytes, inference->cache_hits, inference->cache_misses, generation_started};
                        auto preview = thumbnail(output);
                        {
                            const std::lock_guard lock{mutex};
                            events.push_back({EventKind::generated, request.id, record, std::move(preview), slot, ready, generated});
                            saving[slot] = true;
                            files.push_back({request.id, &output, record, {}, slot});
                        }
                        condition.notify_all();
                        std::println("GENERATE seed={} sample={:.3f}s decode={:.3f}s", request.seed, output.sample_seconds, output.decode_seconds);
                        std::cout.flush();
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
            const std::lock_guard lock{mutex};
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
                error = failure.what();
                paused = true;
                if (task.output) saving[task.slot] = false;
            }
            condition.notify_all();
            glfwPostEmptyEvent();
        }
    }
}
