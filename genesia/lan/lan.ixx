module;
#include <httplib.h>
#include <genesia/cuda.h>
export module genesia.lan;
import genesia.prompt.presets;
import genesia.runtime.session;
import std;

export namespace genesia::lan {
    struct Server final {
        std::atomic_bool listening{};

        explicit Server(const prompt::Catalog& catalog);
        ~Server();
        bool start(const std::string& address, int port, runtime::Session& session);
        void stop();
        void receive(const runtime::Delivery& delivery);

    private:
        struct Job final {
            std::string token;
            std::uint64_t id;
            runtime::State state{runtime::State::running};
            std::filesystem::path image;
            runtime::PreviewFrame preview;
        };
        runtime::Generate operation;
        std::mutex mutex;
        std::optional<Job> job;
        httplib::Server http;
        std::jthread listener;
    };

    std::shared_ptr<const std::uint8_t> capture_preview(const std::uint8_t* source, int width, int height, ::cuda::stream_ref stream);
#if defined(_WIN32)
    std::string wifi_address();
#endif
    int run(std::span<const std::string_view> arguments);
} // namespace genesia::lan
