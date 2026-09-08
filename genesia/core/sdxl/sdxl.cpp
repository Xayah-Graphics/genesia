module;
#include "kernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <genesia/cuda.h>

module genesia.sdxl;

import std;

namespace genesia::sdxl {
    ImageInput::ImageInput(const ::cuda::stream_ref stream, const std::span<const std::uint8_t> source, const int width, const int height)
        : width{width}, height{height}, pixels{stream, ::cuda::device_default_memory_pool(stream.device()), source.size(), ::cuda::no_init}, latent{stream, ::cuda::device_default_memory_pool(stream.device())} {
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{source.data(), source.size()}, pixels);
        stream.sync();
    }

    Model::Model(const ::cuda::stream_ref stream, const std::filesystem::path& file, const std::filesystem::path& cache_directory) : runtime{stream, cache_directory}, network{Checkpoint{file, runtime, weights}}, vae{network.vae}, training{stream, ::cuda::device_default_memory_pool(stream.device()), 1000, ::cuda::no_init}, checkpoint{file} {
        const ClipWorkspaceLayout layout{77};
        ::cuda::device_buffer<std::byte> workspace{stream, ::cuda::device_default_memory_pool(stream.device()), layout.bytes, ::cuda::no_init};
        const Workspace scratch = layout.view(workspace.data());
        kernels::training_sigmas(stream, training.data());
        network.clip_l.prepare_empty(runtime, scratch);
        network.clip_g.prepare_empty(runtime, scratch);
        stream.sync();
    }

    void Model::encode(ImageInput& image) {
        if (!encoder) {
            Checkpoint source{checkpoint, runtime, weights};
            encoder = std::make_unique<VAEEncoder>(source);
        }
        const auto stream = runtime.stream;
        const auto pool = ::cuda::device_default_memory_pool(stream.device());
        const VAEWorkspaceLayout layout{image.height, image.width};
        ::cuda::device_buffer<std::byte> workspace{stream, pool, layout.bytes, ::cuda::no_init};
        ::cuda::device_buffer<__nv_bfloat16> current{stream, pool, std::size_t(image.width) * image.height * 128, ::cuda::no_init};
        image.latent = ::cuda::device_buffer<float>{stream, pool, std::size_t(image.width / 8) * (image.height / 8) * 4, ::cuda::no_init};
        runtime.begin_preparation();
        encoder->forward({image.latent.data(), 1, image.height / 8, image.width / 8, 4, neural::Scalar::f32}, {image.pixels.data(), 1, image.height, image.width, 3}, {current.data()}, runtime, layout.view(workspace.data()));
        auto operator_workspace = runtime.finish_preparation();
        stream.sync();
    }

    Model::Network::Network(Checkpoint&& checkpoint) : clip_l{checkpoint, false}, clip_g{checkpoint, true}, unet{checkpoint}, vae{checkpoint} {}

    Output::Output(const ::cuda::stream_ref stream, const int w, const int h) : pixels{stream, ::cuda::pinned_default_memory_pool(), std::size_t(w) * h * 3, ::cuda::no_init}, width{w}, height{h}, stream{stream} {}

    Inference::Inference(Model& network, Parameters options, Control& control, Snapshots* snapshots, const ImageInput* source)
        : parameters{std::move(options)}, model{network}, control{control}, snapshots{snapshots}, source{source}, unet_layout{parameters.height / 8, parameters.width / 8, parameters.steps}, vae_layout{parameters.height, parameters.width}, workspace{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device())}, unet{model.runtime.stream, parameters.height / 8, parameters.width / 8}, schedule{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), parameters.steps + 1uz, ::cuda::no_init}, seed{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), 1, ::cuda::no_init}, operator_workspace{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device())}, latent{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), std::size_t(parameters.width / 8) * (parameters.height / 8) * 4, ::cuda::no_init},
          step{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), 1, ::cuda::no_init}, input{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), latent.size() * 2, ::cuda::no_init}, epsilon{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), latent.size() * 2, ::cuda::no_init}, decoder{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), std::size_t(parameters.width) * parameters.height * 256, ::cuda::no_init}, decoded{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), std::size_t(parameters.width) * parameters.height * 3, ::cuda::no_init}, image{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), decoded.size(), ::cuda::no_init}, outputs{Output{model.runtime.stream, parameters.width, parameters.height}, Output{model.runtime.stream, parameters.width, parameters.height}} {
        const auto started = std::chrono::steady_clock::now();
        const auto stream  = model.runtime.stream;
        const auto pool    = ::cuda::device_default_memory_pool(stream.device());
        if (source && parameters.denoise == 0) {
            std::size_t free, total;
            neural::check(cudaMemGetInfo(&free, &total));
            resident_bytes = total - free;
            prepare_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            return;
        }
        // The sampling and decoder phases reuse the same activation storage.
        auto* activation                   = reinterpret_cast<__half*>(decoder.data());
        const std::size_t current_elements = 2uz * (parameters.height / 8) * (parameters.width / 8) * 640;
        unet.current                       = {activation, current_elements};
        unet.sequence                      = {activation + current_elements, current_elements / 2};
        model.runtime.begin_preparation();
        const Tokens positive = model.tokenizer.encode(parameters.positive);
        const Tokens negative = model.tokenizer.encode(parameters.negative);
        const ClipWorkspaceLayout clip_layout{positive.ids.size() + negative.ids.size()};
        workspace                      = ::cuda::device_buffer<std::byte>{stream, pool, std::max({clip_layout.bytes, unet_layout.bytes, vae_layout.bytes}), ::cuda::no_init};
        const Workspace text_workspace = clip_layout.view(workspace.data());
        const auto l                   = model.network.clip_l.encode(positive, negative, model.runtime, text_workspace);
        const auto g                   = model.network.clip_g.encode(positive, negative, model.runtime, text_workspace);
        const int length               = std::max(positive.chunks, negative.chunks) * 77;
        ::cuda::device_buffer<__half> context{stream, pool, 2uz * length * 2048, ::cuda::no_init};
        ::cuda::device_buffer<__half> condition{stream, pool, 2 * 2816, ::cuda::no_init};
        kernels::conditioning(stream, context.data(), l.sequence.data(), g.sequence.data(), positive.chunks, negative.chunks, length);
        const std::array<std::int32_t, 2> lengths{positive.chunks * 77, negative.chunks * 77};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::int32_t>{lengths.data(), lengths.size()}, unet.lengths);
        const std::array<float, 6> sizes{float(parameters.height), float(parameters.width), 0, 0, float(parameters.height), float(parameters.width)};
        ::cuda::device_buffer<float> geometry_input{stream, pool, sizes.size(), ::cuda::no_init};
        ::cuda::device_buffer<float> geometry{stream, pool, 1536, ::cuda::no_init};
        ::cuda::device_buffer<float> times{stream, pool, std::size_t(parameters.steps), ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{sizes.data(), sizes.size()}, geometry_input);
        kernels::time_embedding(stream, geometry.data(), geometry_input.data(), 6, 256, 0);
        kernels::conditions(stream, condition.data(), g.pooled.data(), geometry.data());
        kernels::prepare_schedule(stream, schedule.data(), times.data(), model.training.data(), parameters.steps, parameters.denoise);
        model.network.unet.prepare(unet, {context.data(), 2, 1, length, 2048}, {condition.data(), 2, 1, 1, 2816}, times.data(), parameters.steps, model.runtime, unet_layout.view(workspace.data()));
        ::cuda::fill_bytes(stream, seed, 0u);
        kernels::initialize(stream, latent.data(), input.data(), step.data(), seed.data(), schedule.data(), static_cast<int>(latent.size()), source ? source->latent.data() : nullptr, parameters.denoise == 1);
        denoise();
        decode();
        operator_workspace = model.runtime.finish_preparation();
        neural::check(cudaEventCreate(std::out_ptr(initialized)));
        neural::check(cudaEventCreate(std::out_ptr(sampled)));
        neural::check(cudaEventCreate(std::out_ptr(decoded_event)));
        neural::check(cudaGraphCreate(std::out_ptr(graph), 0));
        neural::check(cudaGraphConditionalHandleCreate(&loop, graph.get(), 1, cudaGraphCondAssignDefault));
        neural::check(cudaGraphConditionalHandleCreate(&decode_condition, graph.get(), 0, cudaGraphCondAssignDefault));
        try {
            neural::check(cudaStreamBeginCaptureToGraph(stream.get(), graph.get(), nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
            kernels::initialize(stream, latent.data(), input.data(), step.data(), seed.data(), schedule.data(), static_cast<int>(latent.size()), source ? source->latent.data() : nullptr, parameters.denoise == 1);
            neural::check(cudaEventRecordWithFlags(initialized.get(), stream.get(), cudaEventRecordExternal));
            cudaStreamCaptureStatus status;
            const cudaGraphNode_t* dependencies{};
            std::size_t dependency_count{};
            neural::check(cudaStreamGetCaptureInfo(stream.get(), &status, nullptr, nullptr, &dependencies, nullptr, &dependency_count));
            const std::vector<cudaGraphNode_t> prefix{dependencies, dependencies + dependency_count};
            cudaGraph_t captured{};
            neural::check(cudaStreamEndCapture(stream.get(), &captured));
            cudaGraphNodeParams node{};
            node.type               = cudaGraphNodeTypeConditional;
            node.conditional.handle = loop;
            node.conditional.type   = cudaGraphCondTypeWhile;
            node.conditional.size   = 1;
            cudaGraphNode_t while_node{};
            neural::check(cudaGraphAddNode(&while_node, graph.get(), prefix.data(), nullptr, prefix.size(), &node));
            const cudaGraph_t body = node.conditional.phGraph_out[0];
            neural::check(cudaStreamBeginCaptureToGraph(stream.get(), body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
            denoise();
            kernels::advance(stream, step.data(), parameters.steps, loop, decode_condition, &control);
            neural::check(cudaStreamEndCapture(stream.get(), &captured));
            neural::check(cudaStreamBeginCaptureToGraph(stream.get(), graph.get(), &while_node, nullptr, 1, cudaStreamCaptureModeThreadLocal));
            neural::check(cudaEventRecordWithFlags(sampled.get(), stream.get(), cudaEventRecordExternal));
            neural::check(cudaStreamGetCaptureInfo(stream.get(), &status, nullptr, nullptr, &dependencies, nullptr, &dependency_count));
            const std::vector<cudaGraphNode_t> decode_prefix{dependencies, dependencies + dependency_count};
            neural::check(cudaStreamEndCapture(stream.get(), &captured));
            node.conditional.handle = decode_condition;
            node.conditional.type   = cudaGraphCondTypeIf;
            cudaGraphNode_t decode_node{};
            neural::check(cudaGraphAddNode(&decode_node, graph.get(), decode_prefix.data(), nullptr, decode_prefix.size(), &node));
            neural::check(cudaStreamBeginCaptureToGraph(stream.get(), node.conditional.phGraph_out[0], nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
            decode();
            neural::check(cudaStreamEndCapture(stream.get(), &captured));
            neural::check(cudaStreamBeginCaptureToGraph(stream.get(), graph.get(), &decode_node, nullptr, 1, cudaStreamCaptureModeThreadLocal));
            neural::check(cudaEventRecordWithFlags(decoded_event.get(), stream.get(), cudaEventRecordExternal));
            neural::check(cudaStreamEndCapture(stream.get(), &captured));
        } catch (...) {
            cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
            cudaStreamIsCapturing(stream.get(), &status);
            if (status != cudaStreamCaptureStatusNone) {
                cudaGraph_t captured{};
                cudaStreamEndCapture(stream.get(), &captured);
            }
            throw;
        }
        neural::check(cudaGraphInstantiate(std::out_ptr(executable), graph.get(), 0));
        cache_hits   = model.runtime.cache_hits;
        cache_misses = model.runtime.cache_misses;
        std::size_t free_bytes;
        std::size_t total_bytes;
        neural::check(cudaMemGetInfo(&free_bytes, &total_bytes));
        resident_bytes  = total_bytes - free_bytes;
        prepare_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    Inference::~Inference() {
        model.runtime.stream.sync();
    }

    const Output& Inference::generate(const std::uint64_t seed) {
        Output& result       = outputs[output_index++ % 2];
        result.device_pixels = image.data();
        result.cancelled     = false;
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.completed}.store(0);
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.stage}.store(static_cast<std::uint32_t>(Stage::sampling));
        const auto stream = model.runtime.stream;
        if (source && parameters.denoise == 0) {
            result.sample_seconds = result.decode_seconds = 0;
            ::cuda::copy_bytes(stream, source->pixels, image);
            ::cuda::copy_bytes(stream, image, result.pixels);
            stream.sync();
            result.cancelled = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.cancel}.load() != 0;
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.stage}.store(static_cast<std::uint32_t>(result.cancelled ? Stage::cancelled : Stage::complete));
            return result;
        }
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&seed, 1}, this->seed);
        neural::check(cudaGraphLaunch(executable.get(), stream.get()));
        neural::check(cudaEventSynchronize(decoded_event.get()));
        if (::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.cancel}.load()) {
            result.cancelled = true;
            ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.stage}.store(static_cast<std::uint32_t>(Stage::cancelled));
            return result;
        }
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.stage}.store(static_cast<std::uint32_t>(Stage::transferring));
        ::cuda::copy_bytes(stream, image, result.pixels);
        stream.sync();
        float milliseconds;
        neural::check(cudaEventElapsedTime(&milliseconds, initialized.get(), sampled.get()));
        result.sample_seconds = milliseconds * 0.001;
        neural::check(cudaEventElapsedTime(&milliseconds, sampled.get(), decoded_event.get()));
        result.decode_seconds = milliseconds * 0.001;
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{control.stage}.store(static_cast<std::uint32_t>(Stage::complete));
        return result;
    }

    void Inference::denoise() {
        const int height = parameters.height / 8;
        const int width  = parameters.width / 8;
        model.network.unet.forward({epsilon.data(), 2, height, width, 4}, {input.data(), 2, height, width, 4}, step.data(), unet, model.runtime, unet_layout.view(workspace.data()));
        if (snapshots) kernels::snapshot_begin(model.runtime.stream, snapshots->selected.data(), snapshots->slots.data());
        kernels::euler(model.runtime.stream, latent.data(), input.data(), epsilon.data(), schedule.data(), step.data(), parameters.cfg, height * width * 4, snapshots ? snapshots->latent.data() : nullptr, snapshots ? snapshots->selected.data() : nullptr);
        if (snapshots) kernels::snapshot_publish(model.runtime.stream, snapshots->selected.data(), snapshots->slots.data(), step.data());
    }

    void Inference::decode() {
        model.vae.forward({decoded.data(), 1, parameters.height, parameters.width, 3, neural::Scalar::bf16}, {latent.data(), 1, parameters.height / 8, parameters.width / 8, 4, neural::Scalar::f32}, {decoder.data(), 1, 1, 1, 1, neural::Scalar::bf16}, model.runtime, vae_layout.view(workspace.data()));
        kernels::pixels(model.runtime.stream, image.data(), decoded.data(), parameters.height * parameters.width * 3);
    }
} // namespace genesia::sdxl
