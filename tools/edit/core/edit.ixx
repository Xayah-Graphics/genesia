export module edit.processing;
import std;
export namespace edit {
    inline constexpr std::string_view server_url{"http://127.0.0.1:8000"};
    struct Request final {
        std::filesystem::path input;
        std::string prompt;
    };
    enum class Stage { preparing, editing, stopping };
    inline constexpr std::array<std::string_view, 3> stages{"preparing", "editing", "stopping"};
    struct Progress final {
        Stage stage{};
        std::size_t completed{}, total{};
        std::filesystem::path file;
    };
    struct Result final {
        std::filesystem::path output;
        std::size_t completed{}, total{};
        bool stopped{};
        std::string error;
    };
    bool check_connection();
    Result process(const Request& request, const std::atomic_bool& interrupted, const std::function<void(const Progress&)>& progress);
} // namespace edit
