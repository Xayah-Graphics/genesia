module;
#include <nlohmann/json.hpp>
module genesia.training.state;
import genesia.io.files;
import std;
namespace genesia::training {
    void to_json(nlohmann::json& json, const Config& value) {
        json = {{"physical_batch", value.physical_batch}, {"effective_batch", value.effective_batch}, {"head_only_steps", value.head_only_steps}, {"warmup_steps", value.warmup_steps}, {"head_only_lr", value.head_only_lr}, {"backbone_lr", value.backbone_lr}, {"head_lr", value.head_lr}, {"weight_decay", value.weight_decay}, {"clip_norm", value.clip_norm}, {"seed", value.seed}, {"eval_interval", value.eval_interval}, {"save_interval", value.save_interval}, {"log_interval", value.log_interval}};
    }
    void from_json(const nlohmann::json& json, Config& value) {
        json.at("physical_batch").get_to(value.physical_batch);
        json.at("effective_batch").get_to(value.effective_batch);
        json.at("head_only_steps").get_to(value.head_only_steps);
        json.at("warmup_steps").get_to(value.warmup_steps);
        json.at("head_only_lr").get_to(value.head_only_lr);
        json.at("backbone_lr").get_to(value.backbone_lr);
        json.at("head_lr").get_to(value.head_lr);
        json.at("weight_decay").get_to(value.weight_decay);
        json.at("clip_norm").get_to(value.clip_norm);
        json.at("seed").get_to(value.seed);
        json.at("eval_interval").get_to(value.eval_interval);
        json.at("save_interval").get_to(value.save_interval);
        json.at("log_interval").get_to(value.log_interval);
    }
    void to_json(nlohmann::json& json, const ClassMetric& value) {
        json = {{"name", value.name}, {"precision", value.precision}, {"recall", value.recall}, {"f1", value.f1}, {"samples", value.samples}};
    }
    void from_json(const nlohmann::json& json, ClassMetric& value) {
        json.at("name").get_to(value.name);
        json.at("precision").get_to(value.precision);
        json.at("recall").get_to(value.recall);
        json.at("f1").get_to(value.f1);
        json.at("samples").get_to(value.samples);
    }
    void to_json(nlohmann::json& json, const Metrics& value) {
        json = {{"step", value.step}, {"loss", value.loss}, {"accuracy", value.accuracy}, {"precision", value.precision}, {"recall", value.recall}, {"f1", value.f1}, {"classes", value.classes}, {"confusion", value.confusion}, {"samples", value.samples}};
    }
    void from_json(const nlohmann::json& json, Metrics& value) {
        json.at("step").get_to(value.step);
        json.at("loss").get_to(value.loss);
        json.at("accuracy").get_to(value.accuracy);
        json.at("precision").get_to(value.precision);
        json.at("recall").get_to(value.recall);
        json.at("f1").get_to(value.f1);
        json.at("classes").get_to(value.classes);
        json.at("confusion").get_to(value.confusion);
        json.at("samples").get_to(value.samples);
    }
    void to_json(nlohmann::json& json, const Step& value) {
        json = {{"step", value.step}, {"target", value.target}, {"loss", value.loss}, {"gradient_norm", value.gradient_norm}, {"seconds", value.seconds}};
    }
    void from_json(const nlohmann::json& json, Step& value) {
        json.at("step").get_to(value.step);
        json.at("target").get_to(value.target);
        json.at("loss").get_to(value.loss);
        json.at("gradient_norm").get_to(value.gradient_norm);
        json.at("seconds").get_to(value.seconds);
    }
    void to_json(nlohmann::json& json, const State& value) {
        json = {{"version", 1}, {"fingerprint", value.fingerprint}, {"classes", value.classes}, {"state", phases[std::size_t(value.phase)]}, {"target", value.target}, {"step", value.step}, {"config", value.config}};
    }
    void from_json(const nlohmann::json& json, State& value) {
        if (json.at("version") != 1) throw std::runtime_error{"Unsupported training state format"};
        json.at("fingerprint").get_to(value.fingerprint);
        json.at("classes").get_to(value.classes);
        const auto phase = json.at("state").get<std::string>();
        const auto found = std::ranges::find(phases, phase);
        if (found == phases.end()) throw std::runtime_error{"Unknown training state: " + phase};
        value.phase = Phase(found - phases.begin());
        json.at("target").get_to(value.target);
        json.at("step").get_to(value.step);
        json.at("config").get_to(value.config);
    }
    Config read_config(const std::filesystem::path& path, const Config base) {
        nlohmann::json result = base;
        const auto overrides  = files::read_json(path);
        for (const auto& [key, value] : overrides.items()) result.at(key) = value;
        return result.get<Config>();
    }
    std::optional<State> read_state(const std::filesystem::path& concept_path) {
        const auto root = concept_path / ".genesia" / "training";
        if (!std::filesystem::exists(root / "state.json")) return {};
        auto result       = files::read_json(root / "state.json").get<State>();
        result.checkpoint = std::filesystem::exists(root / "checkpoint.safetensors");
        return result;
    }
    void write_state(const std::filesystem::path& concept_path, const State& state) {
        files::write_json(concept_path / ".genesia" / "training" / "state.json", state);
    }
    History read_history(const std::filesystem::path& concept_path) {
        History result;
        const auto path = concept_path / ".genesia" / "training" / "metrics.jsonl";
        if (!std::filesystem::exists(path)) return result;
        std::ifstream input{path, std::ios::binary};
        input.exceptions(std::ios::badbit);
        std::string line;
        while (std::getline(input, line)) {
            const auto entry = nlohmann::json::parse(line);
            if (entry.at("kind") == "step") result.steps.push_back(entry.at("value").get<Step>());
            else if (entry.at("kind") == "evaluation") result.evaluations.push_back(entry.at("value").get<Metrics>());
            else throw std::runtime_error{"Unknown training metric kind"};
        }
        return result;
    }
    void record_metric(const std::filesystem::path& concept_path, const std::variant<Step, Metrics>& metric) {
        const auto entry = std::visit([]<typename T>(const T& value) { return nlohmann::json{{"kind", std::same_as<T, Step> ? "step" : "evaluation"}, {"value", value}}; }, metric);
        std::ofstream output{concept_path / ".genesia" / "training" / "metrics.jsonl", std::ios::binary | std::ios::app};
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output << entry.dump() << '\n';
    }
} // namespace genesia::training
