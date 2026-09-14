module;
#include <nlohmann/json.hpp>
export module genesia.dataset;
import genesia.hash;
import std;

export namespace genesia::dataset {
    inline const std::filesystem::path directory{GENESIA_DATA_DIRECTORY};
    inline const std::filesystem::path raw             = directory / "raw";
    inline const std::filesystem::path state_directory = directory.parent_path() / ".genesia";

    struct Lock final {
        explicit Lock(std::string_view name, bool wait = true, const std::filesystem::path& directory = state_directory, bool shared = false);
        bool acquired{};
        ~Lock();
        Lock(const Lock&)            = delete;
        Lock& operator=(const Lock&) = delete;

    private:
        std::intptr_t handle{};
    };
    enum class ConceptType { none, classifier, lora };
    inline constexpr std::array<std::string_view, 3> concept_types{"none", "classifier", "lora"};
    struct Concept final {
        std::string key;
        std::filesystem::path path;
        ConceptType type{ConceptType::none};
        bool locked{};
    };
    ConceptType parse_concept_type(std::string_view name);
    void to_json(nlohmann::json& json, const Concept& value);
    Concept read_concept(std::string_view key);
    Concept assign_type(std::string_view key, ConceptType type);

    struct Png final {
        struct Tag final {
            std::string name;
            std::optional<std::string> text;
            float weight{};
        };
        struct Group final {
            bool enabled{};
            std::vector<Tag> tags;
        };
        struct Side final {
            std::string text, fixed;
            std::vector<Group> groups;
        };
        int width{}, height{}, steps{};
        std::uint64_t seed{};
        float cfg{}, denoise{1};
        std::string model, source;
        std::array<Side, 2> prompt;
    };
    Png read_png(const std::filesystem::path& path);

    struct File final {
        std::filesystem::path path;
        std::string entity, sha;
        std::int64_t modified{};
        std::uint64_t bytes{};
        int width{}, height{};
        bool operator==(const File&) const = default;
    };
    struct Collection final {
        std::string key, name;
        std::vector<File> images;
    };
    struct Root final {
        Collection all;
        std::vector<Collection> concepts;
        std::vector<File> files;
        std::vector<std::vector<std::filesystem::path>> conflicts;
        std::string error;
        bool ready{};
    };
    struct Index final {
        std::vector<Root> roots;
        void scan(std::optional<std::string> root = {});
        File identify(const std::filesystem::path& path);

    private:
        struct Cached final {
            std::int64_t modified{};
            std::uint64_t bytes{};
            std::string sha;
            int width{}, height{};
        };
        std::map<std::string, Cached> cache;
    };
} // namespace genesia::dataset
