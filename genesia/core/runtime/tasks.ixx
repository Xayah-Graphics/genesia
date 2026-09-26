export module genesia.runtime.tasks;
export import genesia.data.images;
import std;
export namespace genesia::runtime {
    struct Generate final {
        generation::Settings parameters;
        std::uint64_t seed{};
        int count{1};
        bool random_seed{};
    };
    enum class State { running, saving, complete, stopped, failed };
    inline constexpr std::array<std::string_view, 5> states{"running", "saving", "complete", "stopped", "failed"};
    struct TaskStatus final {
        std::uint64_t id{};
        State state{State::running};
        std::size_t completed{}, total{};
        std::string error;
        bool stopping{};
        std::chrono::steady_clock::time_point started;
        std::shared_ptr<const Generate> request;
    };
    enum class EventKind { generated, saved, task };
    struct GenerationTiming final {
        double sample{}, decode{}, save{};
    };
    struct Event final {
        EventKind kind;
        std::uint64_t id{};
        Record record;
        std::shared_ptr<const void> frame;
        TaskStatus task;
        GenerationTiming timing;
    };
    struct PreviewFrame final {
        std::uint64_t id{};
        std::uint32_t step{};
        int width{}, height{};
        std::shared_ptr<const void> frame;
    };
    enum class GenerationStage { idle, loading, preparing, sampling, decoding, transferring, complete, cancelled };
    struct GenerationProgress final {
        GenerationStage stage{GenerationStage::idle};
        std::uint32_t completed{};
        int steps{};
        bool model_ready{};
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
    };
} // namespace genesia::runtime
