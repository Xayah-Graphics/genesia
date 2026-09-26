export module classifier.progress;
export import classifier.training.state;
import std;
export namespace classifier::runtime {
    struct Stopped final : std::exception {
        const char* what() const noexcept override {
            return "Task stopped";
        }
    };
    enum class Stage { scanning, preparing, audit, classifying, moving, loading_model, preparing_inference };
    inline constexpr std::array<std::string_view, 7> stages{"scanning", "preparing", "audit", "classifying", "moving", "loading model", "preparing inference"};
    struct BatchProgress final {
        Stage stage;
        std::size_t completed{}, total{};
        std::string file;
    };
    struct Progress final {
        std::variant<std::monostate, BatchProgress, training::Step, training::Metrics> value;
    };
} // namespace classifier::runtime
