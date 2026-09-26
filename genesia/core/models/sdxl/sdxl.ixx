module;
#include "kernels.h"
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
    struct Model final {
        Model(::cuda::stream_ref stream, const std::filesystem::path& checkpoint, const std::filesystem::path& cache_directory);

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
        std::size_t cache_hits{};
        std::size_t cache_misses{};
        ::cuda::device_buffer<float> latent;

        Inference(Model& model, generation::Settings parameters, Control& control, Snapshots* snapshots = nullptr);
        ~Inference();
        Inference(const Inference&)            = delete;
        Inference& operator=(const Inference&) = delete;
        Output& sample(std::uint64_t seed);

    private:
        struct Phase final {
            int end;
            Weights::Variant weights;
            UNet network;
            UNetCondition condition;
            cudaGraphConditionalHandle loop{};
        };
        Model& model;
        Control& control;
        Snapshots* snapshots;
        UNetWorkspaceLayout unet_layout;
        ::cuda::device_buffer<std::byte> workspace;
        UNetState unet;
        std::vector<Phase> phases;
        ::cuda::device_buffer<kernels::SamplingStep> schedule;
        ::cuda::device_buffer<std::uint64_t> seed;
        ::cuda::device_buffer<std::byte> operator_workspace;
        ::cuda::device_buffer<int> step;
        ::cuda::device_buffer<__half> input;
        ::cuda::device_buffer<__half> epsilon;
        ::cuda::device_buffer<__half> activations;
        Output output;
        std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)> initialized{nullptr, cudaEventDestroy};
        std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)> sampled{nullptr, cudaEventDestroy};
        std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cudaGraphDestroy)> graph{nullptr, cudaGraphDestroy};
        std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, decltype(&cudaGraphExecDestroy)> executable{nullptr, cudaGraphExecDestroy};

        void denoise(const Phase& phase);
    };
} // namespace genesia::sdxl
