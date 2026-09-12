module;
#include "kernels.h"
#include <cublasLt.h>
#include <cuda_bf16.h>
// cuDNN specializes float JSON as hexadecimal; keep it separate from public metadata.
#define nlohmann cudnn_json
#include <cudnn_frontend.h>
#undef nlohmann
module classifier.network;
import std;
namespace classifier {
    void cuda_check(cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    void blas_check(cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cuBLASLt status {}", int(status)));
    }
    void dnn_check(cudnn_frontend::error_t status) {
        if (status.is_bad()) throw std::runtime_error(status.get_message());
    }
    DeviceBuffer::DeviceBuffer(std::size_t bytes) : size(bytes) {
        if (bytes) cuda_check(cudaMalloc(&data, bytes));
    }
    DeviceBuffer::DeviceBuffer(void* shared, std::size_t bytes) : data(shared), size(bytes), owned(false) {}
    DeviceBuffer::~DeviceBuffer() {
        if (data && owned) cudaFree(data);
    }
    DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : data(std::exchange(other.data, nullptr)), size(std::exchange(other.size, 0)), owned(other.owned) {}
    DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
        if (data && owned) cudaFree(data);
        data  = std::exchange(other.data, nullptr);
        size  = std::exchange(other.size, 0);
        owned = other.owned;
        return *this;
    }
    OptimizerState::OptimizerState(const decltype(SafeFile::header)& metadata) : random_storage(sizeof(RandomState)), update_storage(sizeof(UpdateState)) {
        random = static_cast<RandomState*>(random_storage.data);
        update = static_cast<UpdateState*>(update_storage.data);
        RandomState r{42, std::stoull(metadata.value("sequence", "0"))};
        UpdateState u{};
        u.step = std::stoi(metadata.value("step", "0"));
        cuda_check(cudaMemcpy(random, &r, sizeof(r), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(update, &u, sizeof(u), cudaMemcpyHostToDevice));
    }
    Network::Network(const std::filesystem::path& path, NetworkLoad load) : cache_directory(default_cache_directory()) {
        const bool train = load != NetworkLoad::inference;
        SafeFile file(path);
        if (load == NetworkLoad::pretrained) metadata = {{"architecture", "convnextv2_tiny"}, {"layout", "NHWC-OHWI-DW_RC"}};
        else {
            metadata = file.header.at("__metadata__");
            classes  = decltype(SafeFile::header)::parse(metadata.at("classes").get<std::string>()).get<std::vector<std::string>>();
        }
        for (auto it = file.header.begin(); it != file.header.end(); ++it) {
            if (it.key() == "__metadata__" || it.key().starts_with("optimizer.")) continue;
            Parameter p;
            p.name            = it.key();
            p.shape           = it.value()["shape"].get<std::vector<std::int64_t>>();
            std::size_t count = 1;
            for (auto d : p.shape) count *= d;
            const auto* source = file.base + it.value()["data_offsets"][0].get<std::size_t>();
            const void* values = source;
            std::vector<float> reordered;
            if (load == NetworkLoad::pretrained && p.shape.size() == 4) {
                const auto outputs = p.shape[0], inputs = p.shape[1], height = p.shape[2], width = p.shape[3];
                const auto spatial = height * width;
                reordered.resize(count);
                // timm stores convolution filters as OIHW; the native plans use OHWI and spatial/channel depthwise filters.
                if (p.name.ends_with("conv_dw.weight")) {
                    p.shape = {spatial, outputs};
                    for (std::int64_t r = 0; r < spatial; ++r) for (std::int64_t c = 0; c < outputs; ++c) std::memcpy(&reordered[r * outputs + c], source + (c * spatial + r) * sizeof(float), sizeof(float));
                } else {
                    p.shape = {outputs, height, width, inputs};
                    for (std::int64_t o = 0; o < outputs; ++o) for (std::int64_t r = 0; r < spatial; ++r) for (std::int64_t i = 0; i < inputs; ++i) std::memcpy(&reordered[(o * spatial + r) * inputs + i], source + ((o * inputs + i) * spatial + r) * sizeof(float), sizeof(float));
                }
                values = reordered.data();
            }
            std::size_t fbytes = (count * 4 + 255) / 256 * 256, bbytes = (count * 2 + 255) / 256 * 256;
            p.storage    = DeviceBuffer(fbytes * (train ? 4 : 1) + bbytes);
            p.gpu.master = static_cast<float*>(p.storage.data);
            p.gpu.bf16   = static_cast<std::byte*>(p.storage.data) + fbytes;
            p.gpu.count  = count;
            p.gpu.head   = p.name.starts_with("head.fc.");
            p.gpu.decay  = p.shape.size() > 1 && !p.name.contains("norm") && !p.name.contains("grn");
            cuda_check(cudaMemcpy(p.gpu.master, values, count * 4, cudaMemcpyHostToDevice));
            convert(nullptr, {p.gpu.bf16, 1, 1, 1, int(count), false}, {p.gpu.master, 1, 1, 1, int(count), true});
            if (train) {
                p.gpu.grad = reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + fbytes + bbytes);
                p.gpu.m    = reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + 2 * fbytes + bbytes);
                p.gpu.v    = reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + 3 * fbytes + bbytes);
                cuda_check(cudaMemset(p.gpu.grad, 0, 3 * fbytes));
                if (load == NetworkLoad::resume) for (const auto& state : std::array<std::pair<std::string, float*>, 2>{{{"m", p.gpu.m}, {"v", p.gpu.v}}}) {
                    std::string key = "optimizer." + state.first + "." + p.name;
                    if (file.header.contains(key)) cuda_check(cudaMemcpy(state.second, file.base + file.header[key]["data_offsets"][0].get<std::size_t>(), count * 4, cudaMemcpyHostToDevice));
                }
            }
            parameters.emplace(p.name, std::move(p));
        }
        if (train) optimizer = std::make_unique<OptimizerState>(metadata);
        bind_layers();
    }
    void Network::bind_layers() {
        auto affine             = [&](const std::string& name) { return Affine{&parameters.at(name + ".weight"), &parameters.at(name + ".bias")}; };
        stem                    = affine("stem.0");
        stem_norm               = affine("stem.1");
        constexpr int depths[4] = {3, 3, 9, 3};
        int index               = 0;
        for (int s = 0; s < 4; ++s) {
            auto& stage = stages[s];
            stage.blocks.clear();
            if (s) {
                stage.norm       = affine(std::format("stages.{}.downsample.0", s));
                stage.downsample = affine(std::format("stages.{}.downsample.1", s));
            }
            for (int b = 0; b < depths[s]; ++b, ++index) {
                auto prefix = std::format("stages.{}.blocks.{}", s, b);
                stage.blocks.push_back({affine(prefix + ".conv_dw"), affine(prefix + ".norm"), affine(prefix + ".mlp.fc1"), affine(prefix + ".mlp.grn"), affine(prefix + ".mlp.fc2"), .05f * index / 17});
            }
        }
        head = {affine("head.norm"), affine("head.fc")};
    }
    void Network::initialize_head(const std::vector<std::string>& labels, std::uint64_t seed) {
        std::mt19937 random(static_cast<unsigned>(seed));
        std::normal_distribution<float> normal(0, .02f);
        for (bool bias : std::array{false, true}) {
            auto& p           = parameters.at(bias ? "head.fc.bias" : "head.fc.weight");
            p.shape           = bias ? std::vector<std::int64_t>{std::int64_t(labels.size())} : std::vector<std::int64_t>{std::int64_t(labels.size()), 768};
            std::size_t count = labels.size() * (bias ? 1 : 768), fbytes = (count * 4 + 255) / 256 * 256, bbytes = (count * 2 + 255) / 256 * 256;
            p.storage = DeviceBuffer(4 * fbytes + bbytes);
            p.gpu     = {static_cast<float*>(p.storage.data), static_cast<std::byte*>(p.storage.data) + fbytes, reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + fbytes + bbytes), reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + 2 * fbytes + bbytes), reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + 3 * fbytes + bbytes), count, !bias, true};
            std::vector<float> values(count);
            if (!bias)
                for (float& value : values) value = normal(random);
            cuda_check(cudaMemcpy(p.gpu.master, values.data(), count * 4, cudaMemcpyHostToDevice));
            cuda_check(cudaMemset(p.gpu.grad, 0, 3 * fbytes));
            convert(nullptr, {p.gpu.bf16, 1, 1, 1, int(count), false}, {p.gpu.master, 1, 1, 1, int(count), true});
        }
        classes             = labels;
        metadata["classes"] = decltype(SafeFile::header)(classes).dump();
        bind_layers();
    }
    void Network::save(const std::filesystem::path& path, const decltype(SafeFile::header)& state, bool include_optimizer) {
        cuda_check(cudaDeviceSynchronize());
        std::map<std::string, HostTensor> tensors;
        for (const auto& [name, p] : parameters) {
            HostTensor t{p.shape, std::vector<float>(p.gpu.count)};
            cuda_check(cudaMemcpy(t.values.data(), p.gpu.master, t.values.size() * 4, cudaMemcpyDeviceToHost));
            tensors[name] = std::move(t);
            if (optimizer && include_optimizer)
                for (const auto& s : std::array<std::pair<std::string, float*>, 2>{{{"m", p.gpu.m}, {"v", p.gpu.v}}}) {
                    HostTensor opt{p.shape, std::vector<float>(p.gpu.count)};
                    cuda_check(cudaMemcpy(opt.values.data(), s.second, opt.values.size() * 4, cudaMemcpyDeviceToHost));
                    tensors["optimizer." + s.first + "." + name] = std::move(opt);
                }
        }
        decltype(SafeFile::header) meta = metadata;
        if (optimizer && include_optimizer) {
            RandomState r;
            UpdateState u;
            cuda_check(cudaMemcpy(&r, optimizer->random, sizeof(r), cudaMemcpyDeviceToHost));
            cuda_check(cudaMemcpy(&u, optimizer->update, sizeof(u), cudaMemcpyDeviceToHost));
            meta["step"]           = std::to_string(u.step);
            meta["sequence"]       = std::to_string(r.sequence);
            meta["training_state"] = state.dump();
        } else {
            meta.erase("training_state");
            meta.erase("sequence");
            meta["step"] = "0";
        }
        save_tensors(path, tensors, meta);
    }
    Gemm::~Gemm() {
        if (descriptor) cublasLtMatmulDescDestroy(descriptor);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
    }
    struct TensorLayout {
        std::size_t bytes, maximum;
        std::vector<std::size_t> offsets;
    };
    TensorLayout tensor_layout(const std::vector<Tensor>& tensors, const std::vector<Op>& ops, bool retain) {
        std::size_t maxbytes = 0;
        std::vector<int> last(tensors.size(), 0);
        for (int i = 0; i < int(ops.size()); ++i) {
            last[ops[i].input] = i;
            if (ops[i].skip >= 0) last[ops[i].skip] = i;
        }
        auto layout = [&](bool keep) {
            std::size_t total = 0;
            std::vector<std::size_t> offsets(tensors.size());
            std::vector<std::tuple<int, std::size_t, std::size_t>> live;
            std::vector<std::pair<std::size_t, std::size_t>> free;
            for (int i = 0; i < int(tensors.size()); ++i) {
                auto size = (tensors[i].bytes() + 255) / 256 * 256;
                maxbytes  = std::max(maxbytes, size);
                if (!keep)
                    for (auto it = live.begin(); it != live.end();) {
                        if (std::get<0>(*it) < i - 1) {
                            free.emplace_back(std::get<1>(*it), std::get<2>(*it));
                            it = live.erase(it);
                        } else ++it;
                    }
                auto available = std::ranges::find_if(free, [&](const auto& p) { return p.second >= size; });
                if (available != free.end()) {
                    offsets[i] = available->first;
                    free.erase(available);
                } else {
                    offsets[i] = total;
                    total += size;
                }
                live.emplace_back(last[i], offsets[i], size);
            }
            return std::pair{total, offsets};
        };
        auto [bytes, offsets] = layout(retain);
        return {bytes, maxbytes, std::move(offsets)};
    }
    NetworkPlan::NetworkPlan(Network& m, int n, int w, int h, ActivationStorage lifetime, NetworkPlan* shared) : model(m), batch(n), width(w), height(h), storage(lifetime) {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        blas_check(cublasLtCreate(&blas));
        if (cudnnCreate(&cudnn) != CUDNN_STATUS_SUCCESS) throw std::runtime_error("cudnnCreate failed");
        cudnnSetStream(cudnn, stream);
        tensors.push_back({nullptr, n, h, w, 3, false});
        int x                   = append(Operation::convolution, "stem.0", 0, 96, 4, -1, 0, model.stem);
        x                       = append(Operation::norm, "stem.1", x, 0, 0, -1, 0, model.stem_norm);
        constexpr int depths[4] = {3, 3, 9, 3}, dims[4] = {96, 192, 384, 768};
        for (int stage = 0; stage < 4; ++stage) {
            if (stage) {
                x = append(Operation::norm, std::format("stages.{}.downsample.0", stage), x, 0, 0, -1, 0, model.stages[stage].norm);
                x = append(Operation::convolution, std::format("stages.{}.downsample.1", stage), x, dims[stage], 2, -1, 0, model.stages[stage].downsample);
            }
            for (int j = 0; j < depths[stage]; ++j) {
                auto prefix       = std::format("stages.{}.blocks.{}", stage, j);
                int skip          = x;
                const auto& layer = model.stages[stage].blocks[j];
                x                 = append(Operation::depthwise, prefix + ".conv_dw", x, 0, 0, -1, 0, layer.depthwise);
                x                 = append(Operation::norm, prefix + ".norm", x, 0, 0, -1, 0, layer.norm);
                x                 = append(Operation::linear, prefix + ".mlp.fc1", x, 4 * dims[stage], 0, -1, 0, layer.expand);
                x                 = append(Operation::gelu, prefix + ".mlp.act", x);
                x                 = append(Operation::grn, prefix + ".mlp.grn", x, 0, 0, -1, 0, layer.grn);
                x                 = append(Operation::linear, prefix + ".mlp.fc2", x, dims[stage], 0, -1, 0, layer.project);
                x                 = append(Operation::residual, prefix, x, 0, 0, skip, layer.drop_path);
            }
        }
        x = append(Operation::pool, "head.global_pool", x);
        x = append(Operation::norm, "head.norm", x, 0, 0, -1, 0, model.head.norm);
        x = append(Operation::dropout, "head.drop", x);
        append(Operation::linear, "head.fc", x, int(model.classes.size()), 0, -1, 0, model.head.linear);
        const auto placement = tensor_layout(tensors, ops, storage == ActivationStorage::retain);
        std::size_t bytes = placement.bytes, auxbytes = 0, maxbytes = placement.maximum;
        const auto& offsets = placement.offsets;
        for (auto& op : ops) {
            op.aux_bytes = (op.aux_bytes + 255) / 256 * 256;
            auxbytes += op.aux_bytes;
        }
        arena                       = shared ? DeviceBuffer(shared->arena.data, bytes) : DeviceBuffer(bytes);
        auxiliary                   = shared ? DeviceBuffer(shared->auxiliary.data, auxbytes) : DeviceBuffer(auxbytes);
        std::size_t scratch_bytes   = maxbytes + std::size_t(batch) * 129 * 3072 * 4 + 32 * 1024 * 1024;
        scratch                     = shared ? DeviceBuffer(shared->scratch.data, scratch_bytes) : DeviceBuffer(scratch_bytes);
        std::size_t workspace_bytes = std::size_t(storage == ActivationStorage::retain ? 512 : 128) * 1024 * 1024;
        workspace                   = shared ? DeviceBuffer(shared->workspace.data, workspace_bytes) : DeviceBuffer(workspace_bytes);
        std::size_t pixel_bytes     = std::size_t(batch) * width * height * 3;
        pixels                      = shared ? DeviceBuffer(shared->pixels.data, pixel_bytes) : DeviceBuffer(pixel_bytes);
        auxbytes                    = 0;
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            tensors[i].data = static_cast<std::byte*>(arena.data) + offsets[i];
        }
        for (auto& op : ops) {
            op.aux = reinterpret_cast<float*>(static_cast<std::byte*>(auxiliary.data) + auxbytes);
            auxbytes += op.aux_bytes;
        }
        allocated_bytes = arena.size + auxiliary.size + scratch.size + workspace.size + pixels.size;
        cuda_check(cudaMemsetAsync(arena.data, 0, arena.size, stream));
    }
    NetworkPlan::~NetworkPlan() {
        cudaStreamSynchronize(stream);
        if (inference_exec) cudaGraphExecDestroy(inference_exec);
        if (inference_graph) cudaGraphDestroy(inference_graph);
        convolutions.clear();
        gemms.clear();
        cudnnDestroy(cudnn);
        cublasLtDestroy(blas);
        cudaStreamDestroy(stream);
    }
    int NetworkPlan::append(Operation kind, std::string name, int input, int channels, int kernel, int skip, float probability, Affine affine) {
        Tensor out = tensors[input];
        if (channels) out.c = channels;
        if (kind == Operation::convolution) {
            out.h /= kernel;
            out.w /= kernel;
        }
        if (kind == Operation::pool) {
            out.h = 1;
            out.w = 1;
        }
        out.fp32  = name == "stem.1" || name == "head.norm" || kind == Operation::dropout || (kind == Operation::residual && tensors[skip].fp32);
        int index = int(tensors.size());
        tensors.push_back(out);
        Op op{kind, std::move(name), input, skip, index, kernel, nullptr, nullptr, probability, nullptr, 0};
        if (kind == Operation::convolution || kind == Operation::depthwise || kind == Operation::norm || kind == Operation::linear || kind == Operation::grn) {
            op.weight = affine.weight;
            op.bias   = affine.bias;
        }
        if (kind == Operation::norm) op.aux_bytes = std::size_t(out.n) * out.h * out.w * 2 * 4;
        if (kind == Operation::grn) op.aux_bytes = std::size_t(out.n) * (2 * out.c + 1) * 4;
        if (kind == Operation::residual) op.aux_bytes = out.n * 4;
        if (kind == Operation::dropout) op.aux_bytes = out.elements() * 4;
        ops.push_back(op);
        return index;
    }
    void NetworkPlan::matrix(std::string key, Tensor aa, Tensor bb, Tensor out, bool ta, bool tb, const void* bias, float beta) {
        int ar = aa.n * aa.h * aa.w, br = bb.n * bb.h * bb.w, m = out.n * out.h * out.w, n = out.c, k = ta ? ar : aa.c;
        auto found = gemms.find(key);
        if (found == gemms.end()) {
            auto plan = std::make_unique<Gemm>();
            blas_check(cublasLtMatmulDescCreate(&plan->descriptor, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            cublasOperation_t transa = tb ? CUBLAS_OP_T : CUBLAS_OP_N, transb = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
            blas_check(cublasLtMatmulDescSetAttribute(plan->descriptor, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
            blas_check(cublasLtMatmulDescSetAttribute(plan->descriptor, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
            if (bias) {
                cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
                blas_check(cublasLtMatmulDescSetAttribute(plan->descriptor, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
                blas_check(cublasLtMatmulDescSetAttribute(plan->descriptor, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
            }
            blas_check(cublasLtMatrixLayoutCreate(&plan->a, bb.fp32 ? CUDA_R_32F : CUDA_R_16BF, bb.c, br, bb.c));
            blas_check(cublasLtMatrixLayoutCreate(&plan->b, aa.fp32 ? CUDA_R_32F : CUDA_R_16BF, aa.c, ar, aa.c));
            blas_check(cublasLtMatrixLayoutCreate(&plan->c, out.fp32 ? CUDA_R_32F : CUDA_R_16BF, n, m, n));
            cublasLtMatmulPreference_t preference;
            blas_check(cublasLtMatmulPreferenceCreate(&preference));
            std::size_t limit = workspace.size;
            blas_check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
            std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
            blas_check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction, sizeof(reduction)));
            std::array<cublasLtMatmulHeuristicResult_t, 32> results{};
            int count = 0;
            blas_check(cublasLtMatmulAlgoGetHeuristic(blas, plan->descriptor, plan->a, plan->b, plan->c, plan->c, preference, int(results.size()), results.data(), &count));
            cublasLtMatmulPreferenceDestroy(preference);
            if (!count) throw std::runtime_error("No cuBLASLt algorithm for " + key);
            auto cache = model.cache_directory / "plans" / std::format("lt_v2_{}_{}_{}_{}_{}_{}_{}_{}.json", m, n, k, ta, tb, out.fp32, bias != nullptr, workspace.size);
            if (std::filesystem::exists(cache)) {
                std::ifstream file(cache);
                decltype(SafeFile::header) saved;
                file >> saved;
                auto raw = saved["algorithm"].get<std::vector<std::uint8_t>>();
                std::memcpy(&plan->algorithm, raw.data(), sizeof(plan->algorithm));
                plan->workspace = saved["workspace"];
            } else {
                DeviceBuffer trial(out.bytes());
                cudaEvent_t begin, end;
                cuda_check(cudaEventCreate(&begin));
                cuda_check(cudaEventCreate(&end));
                float best = std::numeric_limits<float>::infinity(), one = 1, zero = 0;
                for (int i = 0; i < count; ++i) {
                    if (results[i].state != CUBLAS_STATUS_SUCCESS) continue;
                    auto status = cublasLtMatmul(blas, plan->descriptor, &one, bb.data, plan->a, aa.data, plan->b, &zero, trial.data, plan->c, trial.data, plan->c, &results[i].algo, workspace.data, results[i].workspaceSize, stream);
                    if (status != CUBLAS_STATUS_SUCCESS) continue;
                    cuda_check(cudaEventRecord(begin, stream));
                    for (int j = 0; j < 3; ++j) blas_check(cublasLtMatmul(blas, plan->descriptor, &one, bb.data, plan->a, aa.data, plan->b, &zero, trial.data, plan->c, trial.data, plan->c, &results[i].algo, workspace.data, results[i].workspaceSize, stream));
                    cuda_check(cudaEventRecord(end, stream));
                    cuda_check(cudaEventSynchronize(end));
                    float elapsed;
                    cuda_check(cudaEventElapsedTime(&elapsed, begin, end));
                    if (elapsed < best) {
                        best            = elapsed;
                        plan->algorithm = results[i].algo;
                        plan->workspace = results[i].workspaceSize;
                    }
                }
                cudaEventDestroy(begin);
                cudaEventDestroy(end);
                std::vector<std::uint8_t> raw(sizeof(plan->algorithm));
                std::memcpy(raw.data(), &plan->algorithm, raw.size());
                write_json(cache, {{"algorithm", raw}, {"workspace", plan->workspace}, {"milliseconds", double(best) / 3}, {"gpu", "RTX 5090 sm_120a"}});
            }
            found = gemms.emplace(key, std::move(plan)).first;
        }
        const auto& p = *found->second;
        float alpha   = 1;
        blas_check(cublasLtMatmul(blas, p.descriptor, &alpha, bb.data, p.a, aa.data, p.b, &beta, out.data, p.c, out.data, p.c, &p.algorithm, workspace.data, p.workspace, stream));
    }
    void NetworkPlan::convolution(std::string key, Tensor x, Parameter& weight, Tensor out, int kernel, int direction, const void* original_input) {
        auto found = convolutions.find(key);
        std::unordered_map<std::int64_t, void*> pointers{{1, x.data}, {2, weight.gpu.bf16}, {3, out.data}};
        if (direction == 0) pointers[4] = model.parameters.at(weight.name.substr(0, weight.name.size() - 6) + "bias").gpu.bf16;
        if (direction == 2) pointers[2] = const_cast<void*>(original_input);
        if (found == convolutions.end()) {
            auto graph = std::make_shared<cudnn_frontend::graph::Graph>();
            auto cache = model.cache_directory / "plans" / std::format("dnn_v2_{}_{}_{}_{}_{}_{}_{}.bin", batch, width, height, key, weight.shape[0], weight.shape[3], workspace.size);
            if (std::filesystem::exists(cache)) {
                Mapping serialized(cache);
                auto begin = static_cast<std::uint8_t*>(serialized.data);
                std::vector<std::uint8_t> bytes(begin, begin + serialized.size);
                dnn_check(graph->deserialize(cudnn, bytes, false, false));
                ConvPlan plan;
                plan.graph     = graph;
                plan.workspace = graph->get_workspace_size();
                found          = convolutions.emplace(key, std::move(plan)).first;
                dnn_check(found->second.graph->execute(cudnn, pointers, workspace.data));
                return;
            }
            graph->set_io_data_type(cudnn_frontend::DataType_t::BFLOAT16).set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT).set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
            auto tx = graph->tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("x").set_uid(1).set_dim({x.n, x.c, x.h, x.w}).set_stride({std::int64_t(x.h) * x.w * x.c, 1, std::int64_t(x.w) * x.c, x.c}));
            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> result;
            if (direction != 2) {
                auto tw = graph->tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("w").set_uid(2).set_dim({weight.shape[0], weight.shape[3], kernel, kernel}).set_stride({weight.shape[1] * weight.shape[2] * weight.shape[3], 1, weight.shape[2] * weight.shape[3], weight.shape[3]}));
                if (direction == 0) {
                    result = graph->conv_fprop(tx, tw, cudnn_frontend::graph::Conv_fprop_attributes{}.set_padding({0, 0}).set_stride({kernel, kernel}).set_dilation({1, 1}));
                    result->set_data_type(cudnn_frontend::DataType_t::BFLOAT16);
                    auto bias = graph->tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("bias").set_uid(4).set_dim({1, out.c, 1, 1}).set_stride({out.c, 1, out.c, out.c}));
                    result    = graph->pointwise(result, bias, cudnn_frontend::graph::Pointwise_attributes{}.set_mode(cudnn_frontend::PointwiseMode_t::ADD));
                } else result = graph->conv_dgrad(tx, tw, cudnn_frontend::graph::Conv_dgrad_attributes{}.set_padding({0, 0}).set_stride({kernel, kernel}).set_dilation({1, 1}));
                result->set_output(true).set_uid(3).set_dim({out.n, out.c, out.h, out.w}).set_stride({std::int64_t(out.h) * out.w * out.c, 1, std::int64_t(out.w) * out.c, out.c});
            } else {
                auto input = graph->tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("input").set_uid(2).set_dim({out.n, out.c, out.h, out.w}).set_stride({std::int64_t(out.h) * out.w * out.c, 1, std::int64_t(out.w) * out.c, out.c}));
                result     = graph->conv_wgrad(tx, input, cudnn_frontend::graph::Conv_wgrad_attributes{}.set_padding({0, 0}).set_stride({kernel, kernel}).set_dilation({1, 1}));
                result->set_output(true).set_uid(3).set_data_type(cudnn_frontend::DataType_t::BFLOAT16).set_dim({weight.shape[0], weight.shape[3], kernel, kernel}).set_stride({weight.shape[1] * weight.shape[2] * weight.shape[3], 1, weight.shape[2] * weight.shape[3], weight.shape[3]});
            }
            dnn_check(graph->validate());
            dnn_check(graph->build_operation_graph(cudnn));
            dnn_check(graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}));
            graph->deselect_workspace_greater_than(workspace.size);
            auto support = graph->check_support();
            if (support.is_bad()) throw std::runtime_error(key + ": " + support.get_message());
            dnn_check(graph->build_plans(cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE));
            ConvPlan plan;
            plan.graph     = graph;
            plan.workspace = graph->get_workspace_size();
            std::vector<std::uint8_t> bytes;
            dnn_check(graph->serialize(bytes));
            std::filesystem::create_directories(cache.parent_path());
            std::ofstream serialized(cache, std::ios::binary);
            serialized.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            found = convolutions.emplace(key, std::move(plan)).first;
        }
        dnn_check(found->second.graph->execute(cudnn, pointers, workspace.data));
    }
    void NetworkPlan::forward(bool stochastic) {
        for (auto& op : ops) {
            Tensor x = tensors[op.input], out = tensors[op.output];
            ParameterView w = op.weight ? op.weight->gpu : ParameterView{}, b = op.bias ? op.bias->gpu : ParameterView{};
            switch (op.kind) {
            case Operation::convolution: convolution(op.name, x, *op.weight, out, op.kernel); break;
            case Operation::depthwise:
                {
                    if (x.fp32) {
                        Tensor bf16 = x;
                        bf16.fp32   = false;
                        bf16.data   = scratch.data;
                        convert(stream, bf16, x);
                        x = bf16;
                    }
                    depthwise(stream, out, x, w.bf16, b.bf16);
                    break;
                }
            case Operation::norm: layer_norm(stream, out, x, w.master, b.master, storage == ActivationStorage::retain ? op.aux : nullptr); break;
            case Operation::linear:
                {
                    if (x.fp32) {
                        Tensor bf16 = x;
                        bf16.fp32   = false;
                        bf16.data   = scratch.data;
                        convert(stream, bf16, x);
                        x = bf16;
                    }
                    matrix(op.name, x, {w.bf16, 1, 1, out.c, x.c, false}, out, false, true, b.bf16);
                    break;
                }
            case Operation::gelu: gelu(stream, out, x); break;
            case Operation::grn: grn(stream, out, x, w.master, b.master, op.aux, static_cast<float*>(scratch.data)); break;
            case Operation::residual:
                {
                    if (storage == ActivationStorage::reuse) add(stream, out, x, tensors[op.skip]);
                    else residual(stream, out, x, tensors[op.skip], op.probability, op.aux, model.optimizer->random, op.output, stochastic);
                    break;
                }
            case Operation::pool: pool(stream, out, x); break;
            case Operation::dropout: dropout(stream, out, x, op.aux, model.optimizer ? model.optimizer->random : nullptr, stochastic); break;
            }
        }
    }
    void NetworkPlan::capture() {
        forward(false);
        cuda_check(cudaStreamSynchronize(stream));
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        forward(false);
        cuda_check(cudaStreamEndCapture(stream, &inference_graph));
        cuda_check(cudaGraphInstantiate(&inference_exec, inference_graph, 0));
    }
    void NetworkPlan::infer(GpuRgb8 input, GpuResult result, cudaStream_t caller_stream, float threshold) {
        preprocess(caller_stream, tensors[0], input.pixels, input.row_stride, std::size_t(input.height) * input.row_stride, nullptr, nullptr, false);
        cuda_check(cudaGraphLaunch(inference_exec, caller_stream));
        softmax(caller_stream, tensors.back(), result.scores, result.indices, result.accepted, threshold);
    }
    TrainingPlan::TrainingPlan(Network& model, int n, int w, int h, TrainingPlan* shared) : NetworkPlan(model, n, w, h, ActivationStorage::retain, shared) {
        const auto placement = tensor_layout(tensors, ops, false);
        gradient_arena       = shared ? DeviceBuffer(shared->gradient_arena.data, placement.bytes) : DeviceBuffer(placement.bytes);
        labels               = shared ? DeviceBuffer(shared->labels.data, n * 4) : DeviceBuffer(n * 4);
        augment              = shared ? DeviceBuffer(shared->augment.data, n * sizeof(Augment)) : DeviceBuffer(n * sizeof(Augment));
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            auto g = tensors[i];
            g.data = static_cast<std::byte*>(gradient_arena.data) + placement.offsets[i];
            gradients.push_back(g);
        }
        UpdateState state;
        cuda_check(cudaMemcpy(&state, model.optimizer->update, sizeof(state), cudaMemcpyDeviceToHost));
        effective_batch = state.effective_batch;
        allocated_bytes += gradient_arena.size + labels.size + augment.size;
    }
    TrainingPlan::~TrainingPlan() {
        cudaStreamSynchronize(stream);
        if (training_exec) cudaGraphExecDestroy(training_exec);
        if (frozen_exec) cudaGraphExecDestroy(frozen_exec);
        if (training_graph) cudaGraphDestroy(training_graph);
        if (frozen_graph) cudaGraphDestroy(frozen_graph);
    }
    void TrainingPlan::backward(bool head_only) {
        cross_entropy(stream, gradients.back(), tensors.back(), static_cast<int*>(labels.data), model.optimizer->update, effective_batch);
        std::vector<bool> ready(tensors.size(), false);
        ready.back() = true;
        for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
            auto& op = *it;
            if (head_only && op.name != "head.fc") break;
            Tensor x = tensors[op.input], dy = gradients[op.output], dx = gradients[op.input];
            bool accumulate = ready[op.input];
            if (accumulate) dx.data = scratch.data;
            float* reduction = reinterpret_cast<float*>(static_cast<std::byte*>(scratch.data) + scratch.size - std::size_t(batch) * 129 * 3072 * 4 - 32 * 1024 * 1024);
            ParameterView w = op.weight ? op.weight->gpu : ParameterView{}, b = op.bias ? op.bias->gpu : ParameterView{};
            switch (op.kind) {
            case Operation::convolution:
                {
                    convolution(op.name + ".dx", dy, *op.weight, dx, op.kernel, 1);
                    Tensor weight_grad{reduction, 1, 1, int(w.count) / x.c, x.c, false};
                    Tensor input_shape = x;
                    input_shape.data   = weight_grad.data;
                    convolution(op.name + ".dw", dy, *op.weight, input_shape, op.kernel, 2, x.data);
                    Tensor accumulated{w.grad, 1, 1, int(w.count) / x.c, x.c, true};
                    add(stream, accumulated, accumulated, weight_grad);
                    bias_backward(stream, dy, b.grad);
                    break;
                }
            case Operation::depthwise: depthwise_backward(stream, dx, dy, x, w.bf16, w.grad, b.grad, reduction); break;
            case Operation::norm: layer_norm_backward(stream, dx, dy, x, w.master, op.aux, w.grad, b.grad, reduction); break;
            case Operation::linear:
                {
                    Tensor input = x;
                    if (input.fp32) {
                        input.data = reduction;
                        input.fp32 = false;
                        convert(stream, input, x);
                    }
                    matrix(op.name + ".dw", dy, input, {w.grad, 1, 1, dy.c, x.c, true}, true, false, nullptr, 1);
                    Tensor native_dx = dx;
                    if (dx.fp32) {
                        native_dx.data = reduction;
                        native_dx.fp32 = false;
                    }
                    matrix(op.name + ".dx", dy, {w.bf16, 1, 1, dy.c, x.c, false}, native_dx, false, false);
                    if (dx.fp32) convert(stream, dx, native_dx);
                    bias_backward(stream, dy, b.grad);
                    break;
                }
            case Operation::gelu: gelu_backward(stream, dx, dy, x); break;
            case Operation::grn: grn_backward(stream, dx, dy, x, w.master, op.aux, w.grad, b.grad, reduction); break;
            case Operation::residual:
                {
                    residual_backward(stream, dx, dy, op.aux);
                    if (ready[op.skip]) add(stream, gradients[op.skip], gradients[op.skip], dy);
                    else convert(stream, gradients[op.skip], dy);
                    ready[op.skip] = true;
                    break;
                }
            case Operation::pool: pool_backward(stream, dx, dy); break;
            case Operation::dropout: dropout_backward(stream, dx, dy, op.aux); break;
            }
            if (accumulate) add(stream, gradients[op.input], gradients[op.input], dx);
            ready[op.input] = true;
        }
    }
    void TrainingPlan::capture_training() {
        capture();
        cuda_check(cudaMemsetAsync(labels.data, 0, labels.size, stream));
        backward(false);
        cuda_check(cudaStreamSynchronize(stream));
        for (auto& [name, p] : model.parameters) cuda_check(cudaMemsetAsync(p.gpu.grad, 0, p.gpu.count * 4, stream));
        for (bool frozen : std::array{true, false}) {
            cudaGraph_t& graph    = frozen ? frozen_graph : training_graph;
            cudaGraphExec_t& exec = frozen ? frozen_exec : training_exec;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            preprocess(stream, tensors[0], static_cast<unsigned char*>(pixels.data), width * 3, std::size_t(width) * height * 3, static_cast<Augment*>(augment.data), model.optimizer->random, true);
            forward(!frozen);
            backward(frozen);
            random_advance(stream, model.optimizer->random);
            cuda_check(cudaStreamEndCapture(stream, &graph));
            cuda_check(cudaGraphInstantiate(&exec, graph, 0));
        }
    }
    void TrainingPlan::microbatch(const unsigned char* host_rgb, const int* host_labels, bool head_only, cudaStream_t caller_stream, cudaEvent_t copied) {
        cuda_check(cudaMemcpyAsync(pixels.data, host_rgb, pixels.size, cudaMemcpyHostToDevice, caller_stream));
        cuda_check(cudaMemcpyAsync(labels.data, host_labels, labels.size, cudaMemcpyHostToDevice, caller_stream));
        cuda_check(cudaEventRecord(copied, caller_stream));
        cuda_check(cudaGraphLaunch(head_only ? frozen_exec : training_exec, caller_stream));
    }
} // namespace classifier
