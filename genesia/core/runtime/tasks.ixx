export module genesia.runtime.tasks;
export import genesia.data.images;
export import genesia.training;
export import genesia.classification.audit;
export import genesia.classification.classify;
export import genesia.runtime.progress;
export import genesia.runtime.catalog;
import std;
export namespace genesia::runtime {
    struct RepaintSource final {
        std::string sha;
        std::filesystem::path path;
    };
    struct Generate final {
        generation::Settings parameters;
        std::uint64_t seed{};
        prompt::Pair prompt;
        std::shared_ptr<const prompt::Catalog> catalog;
        std::optional<RepaintSource> source;
        int count{1};
        bool random_seed{};
    };
    struct Train final {
        training::Options options;
    };
    struct Infer final {
        std::string concept_key;
        std::filesystem::path input;
        std::optional<models::Descriptor> descriptor;
        std::optional<dataset::File> file;
        bool refresh{};
    };
    struct Audit final {
        std::string concept_key;
        bool refresh{};
    };
    struct Fix final {
        std::string concept_key, sha, category;
    };
    struct Undo final {
        std::string concept_key;
    };
    struct Classify final {
        std::string concept_key;
        std::filesystem::path input;
    };
    struct Assign final {
        std::string concept_key;
        dataset::ConceptType type;
    };
    enum class Kind { generate, train, infer, audit, fix, undo, classify, assign };
    inline constexpr std::array<std::string_view, 8> kinds{"generate", "train", "infer", "audit", "fix", "undo", "classify", "assign"};
    struct Request final {
        std::variant<Generate, Train, Infer, Audit, Fix, Undo, Classify, Assign> operation;
    };
    struct Generated final {
        std::filesystem::path path;
        std::uint64_t seed{};
    };
    struct Result final {
        std::variant<std::monostate, Generated, training::State, classification::Result, classification::Audit, dataset::MoveResult, classification::Classification, dataset::Concept> value;
    };
    struct TaskStatus final {
        std::uint64_t id{};
        Kind kind{Kind::generate};
        std::string concept_key;
        State state{State::running};
        std::string image_sha, model_sha;
        Progress progress;
        Result result;
        std::string error;
        bool stopping{};
        std::chrono::steady_clock::time_point started;
        std::shared_ptr<const Request> request;
    };
    enum class EventKind { generated, saved, task };
    struct Event final {
        EventKind kind;
        std::uint64_t id{};
        Record record;
        std::shared_ptr<const void> frame;
        TaskStatus task;
        std::optional<dataset::File> file;
    };
    struct PreviewFrame final {
        std::uint64_t id{};
        std::uint32_t step{};
        int width{}, height{};
        std::shared_ptr<const void> frame;
        bool from_image{};
    };
    enum class GenerationStage { idle, loading, preparing, sampling, decoding, transferring, complete, cancelled };
    struct GenerationProgress final {
        std::uint64_t id{};
        GenerationStage stage{GenerationStage::idle};
        std::uint32_t completed{};
        int steps{};
        bool model_ready{};
        std::chrono::steady_clock::time_point started;
    };
    struct Snapshot final {
        std::optional<TaskStatus> active;
        GenerationProgress generation;
        bool idle{}, finished{}, pending{};
        std::string error;
    };
    struct Delivery final {
        std::deque<Event> events;
        std::deque<PreviewFrame> previews;
        CatalogState catalog;
    };
} // namespace genesia::runtime
