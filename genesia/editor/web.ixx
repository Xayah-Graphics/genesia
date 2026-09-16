module;
#include <genesia/cuda.h>
#include <httplib.h>
export module genesia.editor.web;
import genesia.runtime.tasks;
import std;

export namespace genesia::editor {
    struct Web final {
        std::atomic_bool enabled{}, requested{};
        std::vector<std::string> addresses;

        Web();
        ~Web();
        void start();
        void stop();
        bool watching();
        void update(const runtime::Snapshot& state, std::string unavailable, std::string failure = {});
        void receive(const runtime::Event& event);
        void capture(std::uint64_t task, const std::uint8_t* pixels, int width, int height, ::cuda::stream_ref stream);

    private:
        struct Frame final {
            std::uint64_t task{}, sequence{};
            int width{}, height{};
            std::vector<std::uint8_t> pixels;
            std::filesystem::path file;
        };
        httplib::Server server;
        std::mutex mutex;
        std::condition_variable condition;
        std::chrono::steady_clock::time_point last_seen;
        bool accepting{}, busy{}, closing{};
        std::string status{"等待主机"}, error;
        std::uint64_t task{}, sequence{}, revision{static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())};
        const std::string history_session{std::to_string(revision)};
        std::vector<std::filesystem::path> history;
        std::shared_ptr<const std::string> image;
        std::string mime;
        std::filesystem::path final_image;
        std::optional<Frame> pending;
        std::jthread listener, encoder;
        void encode();
    };
} // namespace genesia::editor
