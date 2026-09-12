module;
#include "kernels.h"
#include <nlohmann/json.hpp>
#include "classifier-audit-report.h"
module classifier.audit;
import classifier.dataset;
import classifier.inference;
import std;

namespace classifier {
    void audit(const std::filesystem::path& dataset, const std::filesystem::path& model, const std::atomic_bool& interrupted) {
        const auto root = std::filesystem::absolute(dataset).lexically_normal();
        const auto model_path = std::filesystem::absolute(model.empty() ? root / "model.safetensors" : model).lexically_normal();
        const auto folders = categories(root);
        std::println("Reading dataset fingerprint: {}", path_utf8(root));
        std::fflush(nullptr);
        const auto content = fingerprint(root);
        Mapping model_file(model_path);
        const auto model_hash = sha256({static_cast<const unsigned char*>(model_file.data), model_file.size});
        std::println("Preparing model: {}", path_utf8(model_path));
        std::fflush(nullptr);
        Pipeline pipeline;
        const std::vector<Descriptor> catalog{{"audit", model_path}};
        const Selection selection{{"audit"}, false};
        pipeline.prepare(catalog, selection, 1024, 1536);
        auto& loaded = pipeline.models.front();
        const auto& network = *loaded.network;
        std::map<std::string, int> labels;
        for (int i = 0; i < int(network.classes.size()); ++i) labels.emplace(network.classes[i], i);
        nlohmann::json provenance, records = nlohmann::json::object();
        if (network.metadata.contains("training_data")) provenance = nlohmann::json::parse(network.metadata.at("training_data").get<std::string>());
        else if (network.metadata.contains("training_state")) {
            const auto state = nlohmann::json::parse(network.metadata.at("training_state").get<std::string>());
            provenance = {{"fingerprint", state.at("fingerprint")}, {"records", state.at("split").at("records")}};
        }
        if (!provenance.is_null()) for (const auto& record : provenance.at("records")) records[record.at("id").get<std::string>()] = record;
        nlohmann::json report{{"version", 1}, {"dataset", path_utf8(root)}, {"fingerprint", content}, {"model", path_utf8(model_path)}, {"model_sha256", model_hash},
            {"step", provenance.is_null() ? nlohmann::json(nullptr) : nlohmann::json(std::stoi(network.metadata.at("step").get<std::string>()))},
            {"training_fingerprint", provenance.is_null() ? nlohmann::json(nullptr) : provenance.at("fingerprint")},
            {"created_utc", std::format("{:%FT%TZ}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()))},
            {"classes", network.classes}, {"threshold", loaded.threshold}, {"rows", nlohmann::json::array()}};
        std::vector<std::pair<std::filesystem::path, int>> paths;
        for (const auto& folder : folders) {
            const auto name = path_utf8(folder.filename());
            const auto label = labels.find(name);
            if (label == labels.end()) throw std::runtime_error("Model has no category: " + name);
            for (const auto& path : image_paths(folder)) paths.emplace_back(path, label->second);
        }
        std::println("Audit: {} images, {} categories; model {}", paths.size(), folders.size(), path_utf8(model_path));
        std::fflush(nullptr);
        for (const auto& [path, label] : paths) {
            if (interrupted.load(std::memory_order_relaxed)) throw std::runtime_error("Audit interrupted; the previous report was retained");
            const auto image = load_image(path);
            const auto shape = std::pair{image.width, image.height};
            if (!loaded.plans.contains(shape)) pipeline.prepare(catalog, selection, image.width, image.height);
            auto& plan = *loaded.plans.at(shape);
            cuda_check(cudaMemcpyAsync(plan.pixels.data, image.rgb.data(), image.rgb.size(), cudaMemcpyHostToDevice, plan.stream));
            auto result = pipeline.run({static_cast<unsigned char*>(plan.pixels.data), image.width, image.height, std::size_t(image.width) * 3}, plan.stream);
            if (!result.error.empty()) throw std::runtime_error(path_utf8(path) + ": " + result.error);
            auto scores = std::move(result.classifiers.front().scores);
            const int predicted = int(std::ranges::max_element(scores) - scores.begin());
            const auto id = path_utf8(path.lexically_relative(root));
            const auto digest = sha256(image.rgb);
            std::string split = "unknown";
            if (!provenance.is_null()) {
                if (!records.contains(id)) split = "new";
                else if (records.at(id).at("sha256") != digest || records.at(id).at("width") != image.width || records.at(id).at("height") != image.height) split = "changed";
                else split = records.at(id).at("split").get<std::string>();
            }
            report["rows"].push_back({{"path", id}, {"label", label}, {"predicted", predicted}, {"scores", scores}, {"split", split}, {"rgb_sha256", digest},
                {"width", image.width}, {"height", image.height}, {"accepted", result.passed}});
            if (report["rows"].size() % 100 == 0 || report["rows"].size() == paths.size()) {
                std::println("Scored {}/{}", report["rows"].size(), paths.size());
                std::fflush(nullptr);
            }
        }
        const auto output = root / ".classifier" / "audit.html";
        std::filesystem::create_directories(output.parent_path());
        auto temporary = output;
        temporary += ".tmp";
        std::string data = report.dump(-1, ' ', true);
        // JSON is embedded in a script element; filenames must never terminate it.
        for (std::size_t i = 0; (i = data.find('<', i)) != std::string::npos; i += 6) data.replace(i, 1, "\\u003c");
        const std::string_view page = audit_report;
        constexpr std::string_view marker = "__GENESIA_AUDIT_DATA__";
        const auto position = page.find(marker);
        std::ofstream file(temporary, std::ios::binary);
        file << page.substr(0, position) << data << page.substr(position + marker.size());
        file.close();
        if (!file) throw std::runtime_error("Cannot write audit report: " + path_utf8(temporary));
        std::filesystem::rename(temporary, output);
        std::println("Report: {}", path_utf8(output));
    }
}
