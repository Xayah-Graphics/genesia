module;
#include "control.h"
#include <cuda_bf16.h>
#include <genesia/cuda.h>

export module genesia.models.sdxl.preview;
import std;
import genesia.compute.inference;
import genesia.models.sdxl.vae;
import genesia.models.sdxl.workspace;

export namespace genesia::sdxl {
    struct Snapshots final {
        const int width;
        const int height;
        ::cuda::host_buffer<SnapshotSlot> slots;
        ::cuda::device_buffer<int> selected;
        ::cuda::device_buffer<float> latent;

        Snapshots(::cuda::stream_ref stream, int width, int height);
    };

    struct Decoder final {
        const int width;
        const int height;
        std::size_t cache_hits{};
        std::size_t cache_misses{};
        ::cuda::stream_ref stream;
        ::cuda::device_buffer<std::uint8_t> pixels;
        std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)> started{nullptr, cudaEventDestroy};
        std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)> finished{nullptr, cudaEventDestroy};

        Decoder(::cuda::stream_ref stream, const VAE& vae, const std::filesystem::path& cache, int width, int height);
        ~Decoder();
        Decoder(const Decoder&)            = delete;
        Decoder& operator=(const Decoder&) = delete;
        void decode(const float* latent);

    private:
        const VAE& vae;
        compute::InferenceRuntime runtime;
        VAEWorkspaceLayout layout;
        ::cuda::device_buffer<float> latent;
        ::cuda::device_buffer<__nv_bfloat16> activations;
        ::cuda::device_buffer<__nv_bfloat16> decoded;
        ::cuda::device_buffer<std::byte> workspace;
        ::cuda::device_buffer<std::byte> operator_workspace;
        std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, decltype(&cudaGraphExecDestroy)> executable{nullptr, cudaGraphExecDestroy};

        void forward();
    };
} // namespace genesia::sdxl
