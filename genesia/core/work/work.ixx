module;
#include "../sdxl/control.h"
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>
export module genesia.work;
export import genesia.generation.output;
export import genesia.prompt;
export import genesia.classifier.training;
export import genesia.classifier.audit;
export import genesia.classifier.classify;
import genesia.generation.defaults;
import genesia.sdxl;
import genesia.sdxl.preview;
import std;

export namespace genesia::work {
    enum class Kind { generate, train, infer, audit, fix, undo, classify };
    struct RepaintSource final {
        std::string sha;
        std::filesystem::path path;
    };
    struct Request final {
        std::uint64_t id{};
        sdxl::Parameters parameters;
        std::uint64_t seed{};
        prompt::Pair prompt;
        std::shared_ptr<const prompt::Catalog> catalog;
        std::optional<RepaintSource> source;
        Kind kind{Kind::generate};
        std::string concept_key, sha, category;
        std::filesystem::path input;
        classifier::TrainingOptions training;
        classifier::Descriptor descriptor;
        std::optional<dataset::File> file;
        std::shared_ptr<dataset::Lock> type_lease;
        bool refresh{}, transient{};
    };
    enum class EventKind { generated, saved, task };
    struct Event final {
        EventKind kind;
        std::uint64_t id;
        Record record;
        std::size_t slot{};
        std::uint64_t ready{};
        nlohmann::json data;
    };
    struct PreviewFrame final {
        std::uint64_t id;
        std::uint32_t step;
        int width, height;
        std::size_t slot;
        std::uint64_t ready;
        bool from_image{};
    };
    struct Visuals final {
        std::function<void()> notify;
        std::function<void(bool, int, int, ::cuda::stream_ref)> prepare;
        std::function<std::uint64_t(bool, const std::uint8_t*, int, int, ::cuda::stream_ref, std::size_t)> publish;
        std::function<bool(std::size_t)> available;
    };
    struct Session final {
        Visuals visuals;
        ::cuda::stream stream;
        ::cuda::host_buffer<sdxl::Control> control;
        std::mutex mutex;
        std::deque<Request> queue;
        std::deque<Event> events;
        std::deque<PreviewFrame> previews;
        std::optional<Request> active;
        std::map<std::uint64_t, nlohmann::json> jobs;
        std::vector<std::string> activated;
        std::chrono::steady_clock::time_point started;
        bool closing{}, worker_done{}, model_ready{};
        bool preview_enabled{defaults::preview_enabled}, preview_visible{true};
        std::string error;

        explicit Session(Visuals visuals = {});
        ~Session();
        std::uint64_t enqueue(Request request);
        void observe(std::vector<Request> requests);
        void cancel(std::uint64_t id);
        void shutdown();

    private:
        struct FileTask final {
            Request request;
            const sdxl::Output* output{};
            Record record;
            std::size_t slot{};
        };
        std::condition_variable condition;
        std::deque<Request> immediate;
        std::deque<FileTask> files;
        std::array<bool, 2> saving{};
        std::atomic_bool interrupted{};
        bool io_closing{}, preview_closing{}, preview_prepare{}, preview_ready{}, preview_working{}, preview_sampling{}, preview_release{}, preview_done{};
        std::shared_ptr<sdxl::Snapshots> snapshots;
        ::cuda::stream preview_stream{::cuda::no_init};
        std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)> preview_finished{nullptr, cudaEventDestroy};
        std::unique_ptr<sdxl::Model> model;
        std::unique_ptr<sdxl::Inference> inference;
        std::unique_ptr<sdxl::ImageInput> source_image;
        std::optional<std::string> encoded_image, prepared_image;
        std::unique_ptr<classifier::Predictions> predictions;
        std::string observed;
        std::uint64_t next_id{1};
        std::size_t iteration{};
        std::jthread worker, io, preview_worker;

        void emit(const Request& request, std::string_view state, nlohmann::json fields = nlohmann::json::object());
        void run();
        void execute(const Request& request);
        void yield();
        bool generate(const Request& request);
        void release_generation();
        void preview_images();
        void write_files();
    };
} // namespace genesia::work
