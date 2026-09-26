export module qwen.session;
export import qwen.processing;
import std;

export namespace qwen::runtime {
    enum class State { idle, running, complete, stopped, failed };
    inline constexpr std::array<std::string_view, 5> states{"idle", "running", "complete", "stopped", "failed"};
    struct Snapshot final {
        std::uint64_t revision{};
        State state{};
        bool busy{}, stopping{};
        Progress progress;
        Result result;
    };
    struct Delivery final {
        Snapshot state;
        std::vector<Progress> progress;
    };
    struct Session final {
        explicit Session(std::function<void()> notify = {});
        ~Session();
        void submit(Request request);
        void cancel();
        Delivery drain();
    private:
        std::function<void()> notify;
        std::mutex mutex;
        std::condition_variable condition;
        std::optional<Request> pending;
        Snapshot state;
        std::vector<Progress> events;
        std::atomic_bool interrupted{};
        bool closing{};
        std::jthread worker;
        void run();
    };
} // namespace qwen::runtime
