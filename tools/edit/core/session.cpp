module edit.session;
import std;
namespace edit::runtime {
    Session::Session(std::function<void()> callback) : notify{std::move(callback)}, worker{[this] { run(); }} {}
    Session::~Session() {
        {
            const std::lock_guard lock{mutex};
            closing     = true;
            interrupted = true;
        }
        condition.notify_one();
        worker.join();
    }
    void Session::submit(Request request) {
        {
            const std::lock_guard lock{mutex};
            if (state.busy) throw std::runtime_error{"Stop the current batch before submitting another"};
            interrupted = false;
            state       = {.revision = state.revision + 1, .state = State::running, .busy = true};
            events.clear();
            pending = std::move(request);
        }
        condition.notify_one();
        if (notify) notify();
    }
    void Session::cancel() {
        {
            const std::lock_guard lock{mutex};
            interrupted    = true;
            state.stopping = state.busy;
            ++state.revision;
        }
        if (notify) notify();
    }
    Delivery Session::drain() {
        const std::lock_guard lock{mutex};
        Delivery delivery{state};
        delivery.progress.swap(events);
        return delivery;
    }
    void Session::run() {
        for (;;) {
            Request request;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this] { return closing || pending.has_value(); });
                if (closing) break;
                request = std::move(*pending);
                pending.reset();
            }
            Result result;
            try {
                result = process(request, interrupted, [this](const Progress& value) {
                    {
                        const std::lock_guard lock{mutex};
                        state.progress = value;
                        events.push_back(value);
                        ++state.revision;
                    }
                    if (notify) notify();
                });
            } catch (const std::exception& error) {
                result.error = error.what();
            }
            {
                const std::lock_guard lock{mutex};
                state.state    = !result.error.empty() ? State::failed : result.stopped ? State::stopped : State::complete;
                state.result   = std::move(result);
                state.busy     = false;
                state.stopping = false;
                ++state.revision;
            }
            if (notify) notify();
        }
    }
} // namespace edit::runtime
