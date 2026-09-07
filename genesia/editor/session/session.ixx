module;
#include <genesia/cuda.h>
#include "../../core/sdxl/control.h"
export module genesia.editor.session;
import genesia.generation.configuration;
import genesia.generation.output;
import genesia.sdxl;
import genesia.editor.platform.interop;
import std;

export namespace genesia::editor {
    struct Request final {
        std::uint64_t id{};
        sdxl::Parameters parameters;
        std::uint64_t seed{};
    };
    enum class EventKind { generated, saved, loaded };
    struct Event final {
        EventKind kind;
        std::uint64_t id;
        Record record;
        Image image;
        std::size_t slot{};
        std::uint64_t ready{};
        std::chrono::steady_clock::time_point generated_at;
    };
    struct Session final {
        const Configuration configuration;
        Interop& interop;
        ::cuda::stream stream;
        ::cuda::host_buffer<sdxl::Control> control;
        std::mutex mutex;
        std::deque<Request> queue;
        std::deque<Event> events;
        std::optional<Request> active;
        std::chrono::steady_clock::time_point started;
        bool paused{};
        bool closing{};
        bool worker_done{};
        bool model_ready{};
        std::string error;

        Session(Configuration configuration, Interop& interop);
        ~Session();
        void enqueue(sdxl::Parameters parameters, std::uint64_t seed);
        void stop();
        void resume();
        void load(std::uint64_t id, std::filesystem::path path);
        void shutdown();

    private:
        struct FileTask final {
            std::uint64_t id;
            const sdxl::Output* output{};
            Record record;
            std::filesystem::path path;
            std::size_t slot{};
        };
        std::condition_variable condition;
        std::deque<FileTask> files;
        std::array<bool, 2> saving{};
        bool io_closing{};
        std::uint64_t next_id{};
        std::jthread worker;
        std::jthread io;

        void generate();
        void write_files();
    };
}
