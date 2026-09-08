module;
#include "control.h"
#include <cuda_bf16.h>
#include <genesia/cuda.h>

export module genesia.sdxl.preview;
import std;
import genesia.neural.inference_runtime;
import genesia.sdxl.vae;
import genesia.sdxl.workspace;

export namespace genesia::sdxl {
    struct Snapshots final {
        const int width;
        const int height;
        ::cuda::host_buffer<SnapshotSlot> slots;
        ::cuda::device_buffer<int> selected;
        ::cuda::device_buffer<float> latent;

        Snapshots(::cuda::stream_ref stream, int width, int height);
    };

    struct Preview final {
        const int width;
        const int height;
        ::cuda::device_buffer<std::uint8_t> pixels;

        Preview(::cuda::stream_ref stream, const VAE& vae, const std::filesystem::path& cache, int width, int height);
        ~Preview();
        Preview(const Preview&)            = delete;
        Preview& operator=(const Preview&) = delete;
        void decode(const float* latent);

    private:
        const VAE& vae;
        neural::InferenceRuntime runtime;
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
