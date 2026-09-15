module;
#include <nlohmann/json.hpp>
export module genesia.data.captions;
export import genesia.data.datasets;
import std;

export namespace genesia::caption {
    struct Document final {
        std::map<std::string, std::vector<std::string>> folders;
        std::set<std::string> bypass;
    };
    struct Folder final {
        std::string path;
        std::size_t direct{}, total{};
        bool excluded{};
    };
    struct Dataset final {
        std::string key;
        Document document;
        std::vector<Folder> folders;
        std::set<std::string> missing_tags;
        std::size_t export_images{};
        std::string issue;
    };
    struct Result final {
        std::string folder;
        std::vector<std::string> tags, effective;
        bool bypass{}, excluded{};
    };
    struct Exported final {
        std::filesystem::path path;
        std::size_t images{};
    };
    void to_json(nlohmann::json& json, const Result& result);
    std::vector<std::string> parse(std::string_view text);
    std::vector<std::string> resolve(const Document& document, std::string_view concept_key, std::string_view folder);
    std::string compose(std::span<const std::string> tags);
    bool excluded(const Document& document, std::string_view folder);
    Dataset inspect(const dataset::Concept& assigned, const dataset::Root& root);
    Result edit(const Dataset& source, std::string_view folder, const std::optional<std::vector<std::string>>& tags, std::optional<bool> bypass);
    Exported export_dataset(const Dataset& source, const dataset::Root& root, const std::filesystem::path& destination, const std::atomic_bool& interrupted, const std::function<std::filesystem::path(const dataset::File&)>& mask, const std::function<void(std::size_t, std::size_t)>& progress);
} // namespace genesia::caption
