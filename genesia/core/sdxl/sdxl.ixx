module;
#include "kernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <genesia/cuda.h>

export module genesia.sdxl;

import std;
import genesia.generation.defaults;
import genesia.neural.inference_runtime;
import genesia.sdxl.tokenizer;
import genesia.sdxl.weights;
import genesia.sdxl.workspace;
import genesia.sdxl.text;
import genesia.sdxl.unet;
import genesia.sdxl.vae;
import genesia.sdxl.preview;

export namespace genesia::sdxl {
    struct Parameters final {
        std::string positive;
        std::string negative;
        int width{defaults::width};
        int height{defaults::height};
        int steps{defaults::steps};
        float cfg{defaults::cfg};
        float denoise{1};
        bool operator==(const Parameters&) const = default;
    };

    struct ImageInput final {
        int width, height;
        ::cuda::device_buffer<std::uint8_t> pixels;
        ::cuda::device_buffer<float> latent;

        ImageInput(::cuda::stream_ref stream, std::span<const std::uint8_t> pixels, int width, int height);
        ImageInput(::cuda::stream_ref stream, int width, int height);
    };

    struct Model final {
        Model(::cuda::stream_ref stream, const std::filesystem::path& checkpoint, const std::filesystem::path& cache_directory);
        void encode(ImageInput& image);

    private:
        friend struct Inference;
        struct Network final {
            Clip clip_l;
            Clip clip_g;
            UNet unet;
            VAE vae;

            explicit Network(Checkpoint&& checkpoint);
        };
        neural::InferenceRuntime runtime;
        Weights weights;
        Network network;
        Tokenizer tokenizer;

    public:
        const VAE& vae;

    private:
        ::cuda::device_buffer<float> training;
        std::filesystem::path checkpoint;
        std::unique_ptr<VAEEncoder> encoder;
    };

    struct Output final {
        ::cuda::host_buffer<std::uint8_t> pixels;
        int width;
        int height;
        double sample_seconds{};
        double decode_seconds{};
        bool cancelled{};
        const std::uint8_t* device_pixels{};
        ::cuda::stream_ref stream;

        Output(::cuda::stream_ref stream, int width, int height);
    };

    struct Inference final {
        const Parameters parameters;
        double prepare_seconds{};
        std::size_t resident_bytes{};
        std::size_t cache_hits{};
        std::size_t cache_misses{};

        Inference(Model& model, Parameters parameters, Control& control, Snapshots* snapshots = nullptr, const ImageInput* source = nullptr, const std::uint8_t* mask = nullptr);
        ~Inference();
        Inference(const Inference&)            = delete;
        Inference& operator=(const Inference&) = delete;
        // Two pinned outputs alternate. Finish reading a result before the
        // second subsequent generate() call reuses its storage.
        const Output& generate(std::uint64_t seed, bool download = true);

    private:
        Model& model;
        Control& control;
        Snapshots* snapshots;
        const ImageInput* source;
        const std::uint8_t* mask;
        UNetWorkspaceLayout unet_layout;
        VAEWorkspaceLayout vae_layout;
        ::cuda::device_buffer<std::byte> workspace;
        UNetState unet;
        ::cuda::device_buffer<kernels::SamplingStep> schedule;
        ::cuda::device_buffer<std::uint64_t> seed;
        ::cuda::device_buffer<std::byte> operator_workspace;
        ::cuda::device_buffer<float> latent;
        ::cuda::device_buffer<float> noise;
        ::cuda::device_buffer<int> step;
        ::cuda::device_buffer<__half> input;
        ::cuda::device_buffer<__half> epsilon;
        ::cuda::device_buffer<__nv_bfloat16> decoder;
        ::cuda::device_buffer<__nv_bfloat16> decoded;
        ::cuda::device_buffer<std::uint8_t> image;
        std::array<Output, 2> outputs;
        std::size_t output_index{};
        std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)> initialized{nullptr, cudaEventDestroy};
        std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)> sampled{nullptr, cudaEventDestroy};
        std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)> decoded_event{nullptr, cudaEventDestroy};
        std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cudaGraphDestroy)> graph{nullptr, cudaGraphDestroy};
        std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, decltype(&cudaGraphExecDestroy)> executable{nullptr, cudaGraphExecDestroy};
        cudaGraphConditionalHandle loop{};
        cudaGraphConditionalHandle decode_condition{};

        void denoise();
        void decode();
    };
} // namespace genesia::sdxl
