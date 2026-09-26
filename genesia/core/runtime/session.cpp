module;
#include <genesia/cuda.h>
module genesia.runtime.session;
import genesia.io.files;
import std;
namespace genesia::runtime {
    Session::Session(generation::Visuals hooks) : visuals{std::move(hooks)}, worker{[this] { run(); }} {}
    Session::~Session() {
        shutdown();
        worker.join();
    }
    TaskStatus Session::submit(Generate request) {
        std::ranges::sort(request.parameters.loras, {}, &generation::Lora::file);
        request.parameters.positive = generation::positive_prompt(request.parameters);
        TaskStatus initial;
        {
            const std::lock_guard lock{mutex};
            if (closing) throw std::runtime_error{"Genesia is closing"};
            if (active) throw std::runtime_error{"Finish the current generation first"};
            const auto count = request.count;
            active           = TaskStatus{.id = next_id++, .total = std::size_t(count), .started = std::chrono::steady_clock::now(), .request = std::make_shared<const Generate>(std::move(request))};
            interrupted      = false;
            submitted        = true;
            initial          = *active;
            delivery.events.push_back({EventKind::task, initial.id, {}, {}, initial});
        }
        condition.notify_one();
        if (visuals.notify) visuals.notify();
        return initial;
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
        const std::lock_guard lock{mutex};
        if (!active || active->id != id) return;
        interrupted      = true;
        active->stopping = true;
        if (generation) generation->cancel();
    }
    void Session::shutdown() {
        {
            const std::lock_guard lock{mutex};
            closing     = true;
            interrupted = true;
            if (generation) generation->cancel();
        }
        condition.notify_one();
    }
    Snapshot Session::snapshot() {
        Snapshot result;
        std::shared_ptr<generation::Engine> engine;
        {
            const std::lock_guard lock{mutex};
            result = {.active = active, .idle = !active, .finished = worker_done, .pending = !delivery.events.empty() || !delivery.previews.empty(), .error = error};
            engine = generation;
        }
        if (engine) result.generation = engine->observe();
        return result;
    }
    Delivery Session::drain() {
        const std::lock_guard lock{mutex};
        return std::exchange(delivery, {});
    }
    void Session::emit(const State state, std::string failure) {
        {
            const std::lock_guard lock{mutex};
            active->state = state;
            active->error = std::move(failure);
            delivery.events.push_back({EventKind::task, active->id, {}, {}, *active});
            if (state >= State::complete) active.reset();
        }
        if (visuals.notify) visuals.notify();
    }
    void Session::receive(Event event) {
        {
            const std::lock_guard lock{mutex};
            if (event.kind == EventKind::task) {
                active->state = event.task.state;
                event.task    = *active;
            } else if (event.kind == EventKind::saved) ++active->completed;
            delivery.events.push_back(std::move(event));
        }
        if (visuals.notify) visuals.notify();
    }
    void Session::run() {
        struct Fingerprint final {
            std::filesystem::file_time_type modified;
            std::uintmax_t bytes{};
            std::string sha;
        };
        std::map<std::string, Fingerprint> fingerprints;
        try {
            for (;;) {
                std::shared_ptr<const Generate> request;
                std::uint64_t id;
                {
                    std::unique_lock lock{mutex};
                    condition.wait(lock, [this] { return closing || submitted; });
                    if (closing) {
                        const bool pending = active.has_value();
                        lock.unlock();
                        if (pending) emit(State::stopped);
                        break;
                    }
                    request   = active->request;
                    id        = active->id;
                    submitted = false;
                }
                try {
                    auto image = *request;
                    for (auto& lora : image.parameters.loras) {
                        const auto path     = std::filesystem::path{project::assets} / "loras" / files::path(lora.file);
                        const auto modified = std::filesystem::last_write_time(path);
                        const auto bytes    = std::filesystem::file_size(path);
                        auto& fingerprint   = fingerprints[lora.file];
                        if (fingerprint.sha.empty() || fingerprint.modified != modified || fingerprint.bytes != bytes) fingerprint = {modified, bytes, files::digest(path)};
                        lora.sha = fingerprint.sha;
                    }
                    std::random_device random;
                    for (int i = 0; i < image.count && !interrupted; ++i) {
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
                                        delivery.previews.push_back(std::move(frame));
                                    }
                                    if (visuals.notify) visuals.notify();
                                });
                            const std::lock_guard lock{mutex};
                            engine->configure(preview_enabled, preview_visible);
                            generation = engine;
                        }
                        image.seed = request->random_seed ? std::uniform_int_distribution<std::uint64_t>{}(random) : request->seed + i;
                        emit(State::running);
                        if (!engine->generate(id, image, interrupted)) break;
                    }
                    emit(interrupted ? State::stopped : State::complete);
                } catch (const std::exception& failure) {
                    emit(State::failed, failure.what());
                }
            }
            std::shared_ptr<generation::Engine> previous;
            {
                const std::lock_guard lock{mutex};
                previous = std::exchange(generation, {});
            }
            if (previous) previous->finish();
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
} // namespace genesia::runtime
