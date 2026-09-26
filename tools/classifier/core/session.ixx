export module classifier.session;
export import classifier.training;
export import classifier.audit;
export import classifier.classify;
import std;
export namespace classifier::runtime {
    struct Load final {
        std::filesystem::path root;
    };
    struct TrainingDetails final {
        std::filesystem::path root;
    };
    struct Train final {
        training::Options options;
    };
    struct Infer final {
        std::filesystem::path root, input;
        bool refresh{};
    };
    struct Audit final {
        std::filesystem::path root;
        bool refresh{};
    };
    struct Fix final {
        std::filesystem::path root;
        std::string sha, category;
    };
    struct Classify final {
        std::filesystem::path root, input;
    };
    struct Recycle final {
        std::filesystem::path root;
        std::string sha;
        std::function<void(const std::filesystem::path&)> file;
    };
    struct Request final {
        std::variant<Load, TrainingDetails, Train, Infer, Audit, Fix, Classify, Recycle> operation;
    };
    enum class Kind { load, training_details, train, infer, audit, fix, classify, recycle };
    enum class State { idle, running, complete, stopped, failed };
    inline constexpr std::array<std::string_view, 5> states{"idle", "running", "complete", "stopped", "failed"};
    struct Inferred final {
        std::filesystem::path path;
        classification::Result prediction;
    };
    struct Result final {
        std::variant<std::monostate, Inferred, dataset::MoveResult, classification::Classification> value;
    };
    struct Snapshot final {
        std::uint64_t revision{};
        Kind kind{Kind::load};
        State state{State::idle};
        bool busy{}, stopping{};
        Progress progress;
        std::string error;
        std::shared_ptr<const models::Info> classifier;
        std::shared_ptr<const training::TrainingData> dataset;
        std::shared_ptr<const training::Info> training;
        std::shared_ptr<const classification::Audit> audit;
        std::shared_ptr<const Result> result;
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
        std::unique_ptr<classification::Predictions> predictions;
        std::jthread worker;
        void progress(const Progress& value);
        void open(const std::filesystem::path& root);
        void read_training(const std::filesystem::path& root);
        std::shared_ptr<const training::TrainingData> scan(const std::filesystem::path& root);
        void train(const training::Options& options);
        void run();
        Result execute(const Request& request);
    };
} // namespace classifier::runtime
