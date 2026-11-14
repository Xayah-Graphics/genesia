module;
#include <httplib.h>
#include <genesia/cuda.h>
export module genesia.lan;
import genesia.runtime.session;
import std;

export namespace genesia::lan {
    inline constexpr int port = 7860;

    struct Server final {
        std::atomic_bool listening{};
        std::atomic_bool requested{};

        Server();
        ~Server();
        bool start(const std::string& address, runtime::Session& session, void (*wake)());
        void stop();
        void complete_submission(std::expected<runtime::TaskStatus, int> submission);
        void receive(const runtime::Delivery& delivery);

    private:
        struct Job final {
            std::string token;
            std::uint64_t id{};
            int steps{}, rejection{};
            runtime::State state{runtime::State::running};
            std::filesystem::path image;
            runtime::PreviewFrame preview;
        };
        std::mutex mutex;
        std::optional<Job> job;
        httplib::Server http;
        std::jthread listener;
    };

    std::shared_ptr<const std::uint8_t> capture_preview(const std::uint8_t* source, int width, int height, const std::shared_ptr<::cuda::stream>& stream);
    std::string wifi_address();
} // namespace genesia::lan
