module;
#include "kernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <genesia/cuda.h>

export module genesia.models.sdxl;

import std;
export import genesia.generation.settings;
import genesia.compute.inference;
import genesia.models.sdxl.tokenizer;
import genesia.models.sdxl.weights;
import genesia.models.sdxl.workspace;
import genesia.models.sdxl.text;
import genesia.models.sdxl.unet;
import genesia.models.sdxl.vae;
import genesia.models.sdxl.preview;

export namespace genesia::sdxl {
    struct ImageInput final {
        int width, height;
        ::cuda::device_buffer<std::uint8_t> pixels;
        ::cuda::device_buffer<float> latent;

        ImageInput(::cuda::stream_ref stream, std::span<const std::uint8_t> pixels, int width, int height);
    };

    struct Model final {
        Model(::cuda::stream_ref stream, const std::filesystem::path& checkpoint, const std::filesystem::path& cache_directory);
        void encode(ImageInput& image);
        void apply_loras(std::span<const generation::Lora> loras);

    private:
        friend struct Inference;
        struct Network final {
            Clip clip_l;
            Clip clip_g;
            UNet unet;
            VAE vae;

            explicit Network(Checkpoint&& checkpoint);
        };
        compute::InferenceRuntime runtime;
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
        const generation::Settings parameters;
        double prepare_seconds{};
        std::size_t resident_bytes{};
        std::size_t cache_hits{};
        std::size_t cache_misses{};

        Inference(Model& model, generation::Settings parameters, Control& control, Snapshots* snapshots = nullptr, const ImageInput* source = nullptr);
        ~Inference();
        Inference(const Inference&)            = delete;
        Inference& operator=(const Inference&) = delete;
        const Output& generate(std::uint64_t seed);

    private:
        Model& model;
        Control& control;
        Snapshots* snapshots;
        const ImageInput* source;
        UNetWorkspaceLayout unet_layout;
        VAEWorkspaceLayout vae_layout;
        ::cuda::device_buffer<std::byte> workspace;
        UNetState unet;
        ::cuda::device_buffer<kernels::SamplingStep> schedule;
        ::cuda::device_buffer<std::uint64_t> seed;
        ::cuda::device_buffer<std::byte> operator_workspace;
        ::cuda::device_buffer<float> latent;
        ::cuda::device_buffer<int> step;
        ::cuda::device_buffer<__half> input;
        ::cuda::device_buffer<__half> epsilon;
        ::cuda::device_buffer<__nv_bfloat16> decoder;
        ::cuda::device_buffer<__nv_bfloat16> decoded;
        ::cuda::device_buffer<std::uint8_t> image;
        Output output;
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
