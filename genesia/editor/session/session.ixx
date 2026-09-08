module;
#include "../../core/sdxl/control.h"
#include <genesia/cuda.h>
export module genesia.editor.session;
import genesia.generation.configuration;
import genesia.generation.output;
import genesia.sdxl;
import genesia.sdxl.preview;
import genesia.editor.platform.interop;
import std;

export namespace genesia::editor {
    struct Request final {
        std::uint64_t id{};
        sdxl::Parameters parameters;
        std::uint64_t seed{};
        prompt::Pair prompt;
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
    struct PreviewFrame final {
        std::uint64_t id;
        std::uint32_t step;
        int width;
        int height;
        std::size_t slot;
        std::uint64_t ready;
    };
    struct Session final {
        const Configuration configuration;
        ImageWriter images;
        Interop& interop;
        Interop& preview_interop;
        ::cuda::stream stream;
        ::cuda::host_buffer<sdxl::Control> control;
        std::mutex mutex;
        std::deque<Request> queue;
        std::deque<Event> events;
        // Bounded by the two external buffers. The UI consumes only the newest image.
        std::deque<PreviewFrame> previews;
        std::optional<Request> active;
        std::chrono::steady_clock::time_point started;
        bool paused{};
        bool closing{};
        bool worker_done{};
        bool model_ready{};
        bool preview_enabled;
        bool preview_visible{true};
        std::string error;

        Session(Configuration configuration, Interop& interop, Interop& preview_interop);
        ~Session();
        void enqueue(sdxl::Parameters parameters, std::uint64_t seed, prompt::Pair prompt);
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
        bool preview_closing{};
        bool preview_prepare{};
        bool preview_ready{};
        bool preview_working{};
        bool preview_sampling{};
        std::shared_ptr<sdxl::Snapshots> snapshots;
        ::cuda::stream preview_stream{::cuda::no_init};
        cudaEvent_t preview_finished{};
        // Both generation and preview graphs reference these immutable model weights.
        std::unique_ptr<sdxl::Model> model;
        std::uint64_t next_id{};
        std::jthread worker;
        std::jthread io;
        std::jthread preview_worker;

        void generate();
        void preview_images();
        void write_files();
    };
} // namespace genesia::editor
