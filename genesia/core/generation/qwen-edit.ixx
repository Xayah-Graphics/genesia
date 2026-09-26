export module genesia.generation.qwen_edit;
export import genesia.classification.inference;
export import genesia.runtime.progress;
import genesia.data.datasets;
import std;

export namespace genesia::qwen {
    struct Profile final {
        std::vector<std::string> classes;
        std::map<std::string, std::string> prompts;
    };
    struct Result final {
        std::filesystem::path output;
        std::size_t completed{}, total{};
        bool stopped{};
        std::string error;
    };
    Profile load_profile(const models::Descriptor& model);
    void save_profile(std::string_view key, const Profile& profile);
    Result fix_images(dataset::Index& index, std::string_view key, const std::filesystem::path& input, const std::map<std::string, std::string>& prompts, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress);
} // namespace genesia::qwen
