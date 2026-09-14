export module genesia.runtime.session;
export import genesia.runtime.tasks;
export import genesia.generation.engine;
import genesia.io.files;
import std;
export namespace genesia::runtime {
    struct Session final {
        explicit Session(generation::Visuals visuals = {});
        ~Session();
        std::uint64_t enqueue(Request request);
        void observe(std::vector<Infer> requests);
        void activate(std::vector<std::string> models);
        void configure_preview(bool enabled, bool visible);
        void cancel(std::uint64_t id);
        void shutdown();
        Snapshot snapshot();
        Delivery drain();

    private:
        struct Task final {
            std::uint64_t id{};
            Request request;
            std::shared_ptr<std::atomic_bool> interrupted{std::make_shared<std::atomic_bool>()};
            std::shared_ptr<files::Lock> type_lease;
        };
        generation::Visuals visuals;
        std::mutex mutex;
        std::condition_variable condition;
        std::deque<Task> queue, immediate;
        std::deque<Event> events;
        std::deque<PreviewFrame> previews;
        std::optional<Task> active, suspended;
        std::map<std::uint64_t, TaskStatus> jobs;
        std::vector<std::string> activated;
        std::shared_ptr<generation::Engine> generation;
        std::unique_ptr<classification::Predictions> predictions;
        bool closing{}, worker_done{}, preview_enabled{defaults::preview_enabled}, preview_visible{true};
        std::string error, observed;
        std::uint64_t next_id{1};
        std::jthread worker;
        void emit(const Task& task, State state, Progress progress = {}, Result result = {}, std::string error = {});
        void receive(Event event);
        void release_generation();
        void run();
        void execute(const Task& task);
        void yield();
    };
} // namespace genesia::runtime
