export module foreground.processing;
import foreground.models.birefnet;
import std;
export namespace foreground {
    struct Request final {
        std::filesystem::path input;
    };
    enum class Stage { scanning, loading_model, masking };
    inline constexpr std::array<std::string_view, 3> stages{"scanning", "loading_model", "masking"};
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
    struct Pipeline final {
        Pipeline();
        ~Pipeline();
        Result process(const Request& request, const std::atomic_bool& interrupted, const std::function<void(const Progress&)>& progress);

    private:
        std::unique_ptr<birefnet::Network> network;
    };
} // namespace foreground
