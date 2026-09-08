module;
#include "kernels.h"
#include <cuda_bf16.h>
#include <genesia/cuda.h>

module genesia.sdxl.preview;
import std;
import genesia.neural.inference_runtime;
import genesia.sdxl.vae;
import genesia.sdxl.workspace;

namespace genesia::sdxl {
    Snapshots::Snapshots(const ::cuda::stream_ref stream, const int w, const int h) : width{w}, height{h}, slots{stream, ::cuda::pinned_default_memory_pool(), 3, ::cuda::no_init}, selected{stream, ::cuda::device_default_memory_pool(stream.device()), 1, ::cuda::no_init}, latent{stream, ::cuda::device_default_memory_pool(stream.device()), 3uz * (w / 8) * (h / 8) * 4, ::cuda::no_init} {
        for (auto& slot : slots) std::construct_at(&slot);
    }

    Preview::Preview(const ::cuda::stream_ref stream, const VAE& vae, const std::filesystem::path& cache, const int w, const int h) : width{w}, height{h}, pixels{stream, ::cuda::device_default_memory_pool(stream.device()), std::size_t(w) * h * 3, ::cuda::no_init}, vae{vae}, runtime{stream, cache}, layout{h, w}, latent{stream, ::cuda::device_default_memory_pool(stream.device()), std::size_t(w / 8) * (h / 8) * 4, ::cuda::no_init}, activations{stream, ::cuda::device_default_memory_pool(stream.device()), std::size_t(w) * h * 256, ::cuda::no_init}, decoded{stream, ::cuda::device_default_memory_pool(stream.device()), std::size_t(w) * h * 3, ::cuda::no_init}, workspace{stream, ::cuda::device_default_memory_pool(stream.device()), layout.bytes, ::cuda::no_init}, operator_workspace{stream, ::cuda::device_default_memory_pool(stream.device())} {
        ::cuda::fill_bytes(stream, latent, 0u);
        forward();
        operator_workspace = runtime.finish_preparation();
        std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cudaGraphDestroy)> graph{nullptr, cudaGraphDestroy};
        neural::check(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
        try {
            forward();
            neural::check(cudaStreamEndCapture(stream.get(), std::out_ptr(graph)));
        } catch (...) {
            cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
            cudaStreamIsCapturing(stream.get(), &status);
            if (status != cudaStreamCaptureStatusNone) cudaStreamEndCapture(stream.get(), std::out_ptr(graph));
            throw;
        }
        neural::check(cudaGraphInstantiate(std::out_ptr(executable), graph.get(), 0));
    }

    Preview::~Preview() {
        runtime.stream.sync();
    }

    void Preview::decode(const float* source) {
        ::cuda::copy_bytes(runtime.stream, ::cuda::std::span<const float>{source, latent.size()}, latent);
        neural::check(cudaGraphLaunch(executable.get(), runtime.stream.get()));
    }

    void Preview::forward() {
        vae.forward({decoded.data(), 1, height, width, 3, neural::Scalar::bf16}, {latent.data(), 1, height / 8, width / 8, 4, neural::Scalar::f32}, {activations.data(), 1, 1, 1, 1, neural::Scalar::bf16}, runtime, layout.view(workspace.data()));
        kernels::pixels(runtime.stream, pixels.data(), decoded.data(), width * height * 3);
    }
} // namespace genesia::sdxl
