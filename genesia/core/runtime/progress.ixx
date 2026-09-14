export module genesia.runtime.progress;
export import genesia.training.state;
import std;
export namespace genesia::runtime {
    struct Stopped final : std::exception {
        const char* what() const noexcept override {
            return "Task stopped";
        }
    };
    enum class State { running, saving, complete, stopped, failed };
    inline constexpr std::array<std::string_view, 5> states{"running", "saving", "complete", "stopped", "failed"};
    enum class Stage { preparing, audit, classifying, moving, generating, scanning, normalizing };
    inline constexpr std::array<std::string_view, 7> stages{"preparing", "audit", "classifying", "moving", "generating", "scanning", "normalizing"};
    struct BatchProgress final {
        Stage stage;
        std::size_t completed{}, total{};
    };
    struct Progress final {
        std::variant<std::monostate, BatchProgress, training::Step, training::Metrics> value;
    };
} // namespace genesia::runtime
