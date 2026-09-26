export module qwen.processing;
import std;

export namespace qwen {
    inline constexpr std::string_view server_url{"http://127.0.0.1:1234"};
    inline constexpr std::string_view model{"qwen3.8"};
    struct Request final {
        std::filesystem::path input;
        std::string prompt;
        std::vector<std::string> choices;
    };
    enum class Stage { scanning, classifying, stopping };
    inline constexpr std::array<std::string_view, 3> stages{"scanning", "classifying", "stopping"};
    struct Progress final {
        Stage stage{};
        std::size_t completed{}, total{};
        std::filesystem::path file;
        std::string result;
    };
    struct Item final {
        std::filesystem::path file;
        std::string result;
    };
    struct Result final {
        std::vector<Item> items;
        std::size_t total{};
        bool stopped{};
        std::string error;
    };
    bool check_connection();
    Result process(const Request& request, std::atomic_bool& interrupted, const std::function<void(const Progress&)>& progress);
} // namespace qwen
