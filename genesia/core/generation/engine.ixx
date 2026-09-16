module;
#include "../models/sdxl/control.h"
#include <genesia/cuda.h>
export module genesia.generation.engine;
export import genesia.runtime.tasks;
import genesia.models.sdxl;
import genesia.models.sdxl.preview;
import genesia.generation.output;
import std;
export namespace genesia::generation {
    struct Visuals final {
        std::function<void()> notify;
        std::function<void(bool, int, int, ::cuda::stream_ref)> prepare;
        std::function<std::shared_ptr<const void>(std::uint64_t, bool, const std::uint8_t*, int, int, ::cuda::stream_ref, std::size_t)> publish;
        std::function<bool(std::size_t)> available;
    };
    struct SavedImage final {
        Record record;
        dataset::File file;
        std::span<const std::uint8_t> pixels;
    };
    struct Engine final {
        Engine(Visuals visuals, std::function<void(runtime::Event)> report, std::function<void(runtime::PreviewFrame)> preview);
        ~Engine();
        std::optional<SavedImage> generate(dataset::Index& index, std::uint64_t id, const runtime::Generate& request, const std::atomic_bool& interrupted);
        void cancel();
        void configure(bool enabled, bool visible);
        runtime::GenerationProgress observe();
        void finish();

    private:
        struct Active final {
            std::uint64_t id;
            runtime::Generate request;
        };
        Visuals visuals;
        std::function<void(runtime::Event)> report;
        std::function<void(runtime::PreviewFrame)> preview;
        ::cuda::stream stream;
        ::cuda::host_buffer<sdxl::Control> control;
        std::mutex mutex;
        std::condition_variable condition;
        std::optional<Active> active;
        std::chrono::steady_clock::time_point started;
        bool model_ready{}, preview_enabled{defaults::preview_enabled}, preview_visible{true};
        bool finished{}, preview_closing{}, preview_prepare{}, preview_ready{}, preview_working{}, preview_sampling{}, preview_release{}, preview_done{};
        std::string error;
        std::shared_ptr<sdxl::Snapshots> snapshots;
        ::cuda::stream preview_stream{::cuda::no_init};
        std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)> preview_finished{nullptr, cudaEventDestroy};
        std::unique_ptr<sdxl::Model> model;
        std::unique_ptr<sdxl::Inference> inference;
        std::unique_ptr<sdxl::ImageInput> source_image;
        std::optional<std::string> encoded_image, prepared_image;
        std::size_t iteration{};
        std::jthread preview_worker;
        void release();
        void preview_images();
    };
} // namespace genesia::generation
