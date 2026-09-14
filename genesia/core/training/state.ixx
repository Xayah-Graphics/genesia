module;
#include <nlohmann/json.hpp>
export module genesia.training.state;
import std;
export namespace genesia::training {
    struct Config final {
        int physical_batch{4}, effective_batch{64}, head_only_steps{25}, warmup_steps{10};
        float head_only_lr{.001f}, backbone_lr{.00003f}, head_lr{.0003f}, weight_decay{.01f}, clip_norm{1};
        std::uint64_t seed{42};
        int eval_interval{25}, save_interval{25}, log_interval{5};
        bool operator==(const Config&) const = default;
    };
    struct ClassMetric final {
        std::string name;
        double precision{}, recall{}, f1{};
        std::size_t samples{};
    };
    struct Metrics final {
        int step{};
        double loss{}, accuracy{}, precision{}, recall{}, f1{};
        std::vector<ClassMetric> classes;
        std::vector<std::vector<int>> confusion;
        std::size_t samples{};
    };
    struct Step final {
        int step{}, target{};
        float loss{}, gradient_norm{};
        double seconds{};
    };
    enum class Phase { preparing, training, stopped, complete, failed };
    inline constexpr std::array<std::string_view, 5> phases{"preparing", "training", "stopped", "complete", "failed"};
    struct State final {
        std::string fingerprint;
        std::vector<std::string> classes;
        Phase phase{Phase::preparing};
        int target{}, step{};
        Config config;
        bool checkpoint{};
    };
    struct History final {
        std::vector<Step> steps;
        std::vector<Metrics> evaluations;
    };
    void to_json(nlohmann::json& json, const Config& value);
    void from_json(const nlohmann::json& json, Config& value);
    void to_json(nlohmann::json& json, const ClassMetric& value);
    void from_json(const nlohmann::json& json, ClassMetric& value);
    void to_json(nlohmann::json& json, const Metrics& value);
    void from_json(const nlohmann::json& json, Metrics& value);
    void to_json(nlohmann::json& json, const Step& value);
    void from_json(const nlohmann::json& json, Step& value);
    void to_json(nlohmann::json& json, const State& value);
    void from_json(const nlohmann::json& json, State& value);
    Config read_config(const std::filesystem::path& path, Config base = {});
    std::optional<State> read_state(const std::filesystem::path& concept_path);
    void write_state(const std::filesystem::path& concept_path, const State& state);
    History read_history(const std::filesystem::path& concept_path);
    void record_metric(const std::filesystem::path& concept_path, const std::variant<Step, Metrics>& metric);
} // namespace genesia::training
