export module genesia.runtime.session;
export import genesia.runtime.tasks;
export import genesia.generation.engine;
import std;
export namespace genesia::runtime {
    struct Session final {
        explicit Session(generation::Visuals visuals = {});
        ~Session();
        TaskStatus submit(Generate request);
        void configure_preview(bool enabled, bool visible);
        void cancel(std::uint64_t id);
        void shutdown();
        Snapshot snapshot();
        Delivery drain();

    private:
        generation::Visuals visuals;
        std::mutex mutex;
        std::condition_variable condition;
        Delivery delivery;
        std::optional<TaskStatus> active;
        std::shared_ptr<generation::Engine> generation;
        std::atomic_bool interrupted{};
        bool closing{}, worker_done{}, submitted{}, preview_enabled{defaults::preview_enabled}, preview_visible{true};
        std::string error;
        std::uint64_t next_id{1};
        std::jthread worker;
        void emit(State state, std::string failure = {});
        void receive(Event event);
        void run();
    };
} // namespace genesia::runtime
