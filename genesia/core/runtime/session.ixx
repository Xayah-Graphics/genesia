export module genesia.runtime.session;
export import genesia.runtime.tasks;
export import genesia.generation.engine;
import std;
export namespace genesia::runtime {
    struct Session final {
        explicit Session(generation::Visuals visuals = {}, bool browse = false);
        ~Session();
        TaskStatus submit(Request request);
        void select(std::string key);
        void observe(std::vector<Infer> requests, std::vector<dataset::File> masks, bool generated_masks);
        void activate(std::vector<std::string> models);
        void configure_preview(bool enabled, bool visible);
        void cancel(std::uint64_t id);
        void shutdown();
        Snapshot snapshot();
        Delivery drain();

    private:
        generation::Visuals visuals;
        Catalog catalog;
        std::mutex mutex;
        std::condition_variable condition;
        Delivery delivery;
        std::optional<TaskStatus> active;
        std::vector<Infer> wanted;
        std::vector<dataset::File> wanted_masks;
        foreground::Pipeline foreground;
        bool generated_masks{};
        std::vector<std::string> activated;
        std::shared_ptr<generation::Engine> generation;
        std::unique_ptr<classification::Predictions> predictions;
        std::atomic_bool interrupted{};
        bool closing{}, worker_done{}, loading{}, submitted{}, inferring{}, preview_enabled{defaults::preview_enabled}, preview_visible{true};
        std::string error, observed, selection, inspect_key;
        std::uint64_t next_id{1};
        std::jthread worker;
        void emit(State state, Progress progress = {}, Result result = {}, std::string failure = {});
        void receive(Event event);
        void update_catalog(CatalogState update);
        void moved(const dataset::MoveResult& movement);
        void release_generation();
        void run(bool browse);
        void execute(const Request& request);
        TaskStatus infer(Infer request, std::span<const std::uint8_t> rgb = {});
        TaskStatus segment(Mask request, std::span<const std::uint8_t> rgb = {});
    };
} // namespace genesia::runtime
