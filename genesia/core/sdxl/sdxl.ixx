module;
#include "kernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <genesia/cuda.h>

export module genesia.sdxl;

import std;
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
        int width{1024};
        int height{1536};
        int steps{50};
        float cfg{4.5F};
        bool operator==(const Parameters&) const = default;
    };

    struct Model final {
        Model(::cuda::stream_ref stream, const std::filesystem::path& checkpoint, const std::filesystem::path& cache_directory);

    private:
        friend struct Inference;
        neural::InferenceRuntime runtime;
        Weights weights;
        Checkpoint checkpoint;
        Tokenizer tokenizer;
        Clip clip_l;
        Clip clip_g;
        UNet unet;

    public:
        const VAE vae;

    private:
        ::cuda::device_buffer<float> training;
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
        double text_seconds{};
        double precompute_seconds{};
        double tuning_seconds{};

        Inference(Model& model, Parameters parameters, Control& control, Snapshots* snapshots = nullptr);
        ~Inference();
        Inference(const Inference&)            = delete;
        Inference& operator=(const Inference&) = delete;
        // Two pinned outputs alternate. Finish reading a result before the
        // second subsequent generate() call reuses its storage.
        const Output& generate(std::uint64_t seed);

    private:
        Model& model;
        Control& control;
        Snapshots* snapshots;
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
        std::array<Output, 2> outputs;
        std::size_t output_index{};
        cudaGraph_t graph{};
        cudaGraphExec_t executable{};
        cudaGraphConditionalHandle loop{};
        cudaGraphConditionalHandle decode_condition{};
        cudaEvent_t initialized{};
        cudaEvent_t sampled{};
        cudaEvent_t decoded_event{};

        void denoise();
        void decode();
    };
} // namespace genesia::sdxl
