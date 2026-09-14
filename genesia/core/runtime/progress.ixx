export module genesia.runtime.progress;
export import genesia.training.state;
import std;
export namespace genesia::runtime {
    struct Stopped final : std::exception {
        const char* what() const noexcept override {
            return "Task stopped";
        }
    };
    enum class State { queued, running, saving, complete, stopped, failed };
    inline constexpr std::array<std::string_view, 6> states{"queued", "running", "saving", "complete", "stopped", "failed"};
    enum class Stage { preparing, audit, classifying, moving };
    inline constexpr std::array<std::string_view, 4> stages{"preparing", "audit", "classifying", "moving"};
    struct BatchProgress final {
        Stage stage;
        std::size_t completed{}, total{};
    };
    struct Progress final {
        std::variant<std::monostate, BatchProgress, training::Step, training::Metrics> value;
    };
} // namespace genesia::runtime
