module;
#include "../../compute/tensor.h"
#include "kernels.h"
#include <cublasLt.h>
#include <cuda_bf16.h>
// cuDNN specializes float JSON as hexadecimal; keep it separate from public metadata.
#define nlohmann cudnn_json
#include <cudnn_frontend.h>
#undef nlohmann
module genesia.models.convnext;
import std;
namespace genesia::convnext {
    void dnn_check(cudnn_frontend::error_t status) {
        if (status.is_bad()) throw std::runtime_error(status.get_message());
    }
    OptimizerState::OptimizerState() : random_storage(sizeof(RandomState)), update_storage(sizeof(UpdateState)) {
        random = static_cast<RandomState*>(random_storage.data);
        update = static_cast<UpdateState*>(update_storage.data);
    }
    Network::Network(const std::filesystem::path& path, NetworkLoad load) : cache_directory(compute::plan_directory(std::filesystem::path{GENESIA_CACHE_DIRECTORY}, 0)) {
        const bool train = load != NetworkLoad::inference;
        files::SafeFile file(path);
        if (load != NetworkLoad::pretrained) {
            const auto& metadata = file.header.at("__metadata__");
            if (metadata.at("format") != "genesia-convnext-2") throw std::runtime_error{"Unsupported classifier model format"};
            classes = decltype(files::SafeFile::header)::parse(metadata.at("classes").get<std::string>()).get<std::vector<std::string>>();
            step    = std::stoi(metadata.at("step").get<std::string>());
            if (load == NetworkLoad::resume) sequence = std::stoull(metadata.at("sequence").get<std::string>());
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
                    for (std::int64_t r = 0; r < spatial; ++r)
                        for (std::int64_t c = 0; c < outputs; ++c) std::memcpy(&reordered[r * outputs + c], source + (c * spatial + r) * sizeof(float), sizeof(float));
                } else {
                    p.shape = {outputs, height, width, inputs};
                    for (std::int64_t o = 0; o < outputs; ++o)
                        for (std::int64_t r = 0; r < spatial; ++r)
                            for (std::int64_t i = 0; i < inputs; ++i) std::memcpy(&reordered[(o * spatial + r) * inputs + i], source + ((o * inputs + i) * spatial + r) * sizeof(float), sizeof(float));
                }
                values = reordered.data();
            }
            std::size_t fbytes = (count * 4 + 255) / 256 * 256, bbytes = (count * 2 + 255) / 256 * 256;
            p.storage    = compute::DeviceBuffer(fbytes * (train ? 4 : 1) + bbytes);
            p.gpu.master = static_cast<float*>(p.storage.data);
            p.gpu.bf16   = static_cast<std::byte*>(p.storage.data) + fbytes;
            p.gpu.count  = count;
            p.gpu.head   = p.name.starts_with("head.fc.");
            p.gpu.decay  = p.shape.size() > 1 && !p.name.contains("norm") && !p.name.contains("grn");
            compute::check(cudaMemcpy(p.gpu.master, values, count * 4, cudaMemcpyHostToDevice));
            convert(nullptr, {p.gpu.bf16, 1, 1, 1, int(count), compute::Scalar::bf16}, {p.gpu.master, 1, 1, 1, int(count), compute::Scalar::f32});
            if (train) {
                p.gpu.grad = reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + fbytes + bbytes);
                p.gpu.m    = reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + 2 * fbytes + bbytes);
                p.gpu.v    = reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + 3 * fbytes + bbytes);
                compute::check(cudaMemset(p.gpu.grad, 0, 3 * fbytes));
                if (load == NetworkLoad::resume)
                    for (const auto& state : std::array<std::pair<std::string, float*>, 2>{{{"m", p.gpu.m}, {"v", p.gpu.v}}}) {
                        std::string key = "optimizer." + state.first + "." + p.name;
                        compute::check(cudaMemcpy(state.second, file.base + file.header.at(key).at("data_offsets")[0].get<std::size_t>(), count * 4, cudaMemcpyHostToDevice));
                    }
            }
            parameters.emplace(p.name, std::move(p));
        }
        if (train) optimizer = std::make_unique<OptimizerState>();
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
            p.storage = compute::DeviceBuffer(4 * fbytes + bbytes);
            p.gpu     = {static_cast<float*>(p.storage.data), static_cast<std::byte*>(p.storage.data) + fbytes, reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + fbytes + bbytes), reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + 2 * fbytes + bbytes), reinterpret_cast<float*>(static_cast<std::byte*>(p.storage.data) + 3 * fbytes + bbytes), count, !bias, true};
            std::vector<float> values(count);
            if (!bias)
                for (float& value : values) value = normal(random);
            compute::check(cudaMemcpy(p.gpu.master, values.data(), count * 4, cudaMemcpyHostToDevice));
            compute::check(cudaMemset(p.gpu.grad, 0, 3 * fbytes));
            convert(nullptr, {p.gpu.bf16, 1, 1, 1, int(count), compute::Scalar::bf16}, {p.gpu.master, 1, 1, 1, int(count), compute::Scalar::f32});
        }
        classes = labels;
        bind_layers();
    }
    void Network::save(const std::filesystem::path& path, std::optional<std::string_view> training_state) {
        compute::check(cudaDeviceSynchronize());
        std::map<std::string, files::HostTensor> tensors;
        for (const auto& [name, p] : parameters) {
            files::HostTensor t{p.shape, std::vector<float>(p.gpu.count)};
            compute::check(cudaMemcpy(t.values.data(), p.gpu.master, t.values.size() * 4, cudaMemcpyDeviceToHost));
            tensors[name] = std::move(t);
            if (optimizer && training_state)
                for (const auto& s : std::array<std::pair<std::string, float*>, 2>{{{"m", p.gpu.m}, {"v", p.gpu.v}}}) {
                    files::HostTensor opt{p.shape, std::vector<float>(p.gpu.count)};
                    compute::check(cudaMemcpy(opt.values.data(), s.second, opt.values.size() * 4, cudaMemcpyDeviceToHost));
                    tensors["optimizer." + s.first + "." + name] = std::move(opt);
                }
        }
        decltype(files::SafeFile::header) meta{{"format", "genesia-convnext-2"}, {"architecture", "convnextv2_tiny"}, {"layout", "NHWC-OHWI-DW_RC"}, {"classes", decltype(files::SafeFile::header)(classes).dump()}, {"step", std::to_string(step)}};
        if (optimizer) {
            UpdateState u;
            compute::check(cudaMemcpy(&u, optimizer->update, sizeof(u), cudaMemcpyDeviceToHost));
            meta["step"] = std::to_string(u.step);
        }
        if (optimizer && training_state) {
            RandomState r;
            compute::check(cudaMemcpy(&r, optimizer->random, sizeof(r), cudaMemcpyDeviceToHost));
            meta["sequence"]       = std::to_string(r.sequence);
            meta["training_state"] = *training_state;
        }
        files::save_tensors(path, tensors, meta);
    }
    struct TensorLayout {
        std::size_t bytes, maximum;
        std::vector<std::size_t> offsets;
    };
    TensorLayout tensor_layout(const std::vector<compute::TensorView>& tensors, const std::vector<Op>& ops, bool retain) {
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
        compute::check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        handles.emplace(stream);
        tensors.push_back({nullptr, n, h, w, 3, compute::Scalar::bf16});
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
        arena                       = shared ? compute::DeviceBuffer(shared->arena.data, bytes) : compute::DeviceBuffer(bytes);
        auxiliary                   = shared ? compute::DeviceBuffer(shared->auxiliary.data, auxbytes) : compute::DeviceBuffer(auxbytes);
        std::size_t scratch_bytes   = maxbytes + std::size_t(batch) * 129 * 3072 * 4 + 32 * 1024 * 1024;
        scratch                     = shared ? compute::DeviceBuffer(shared->scratch.data, scratch_bytes) : compute::DeviceBuffer(scratch_bytes);
        std::size_t workspace_bytes = std::size_t(storage == ActivationStorage::retain ? 512 : 128) * 1024 * 1024;
        workspace                   = shared ? compute::DeviceBuffer(shared->workspace.data, workspace_bytes) : compute::DeviceBuffer(workspace_bytes);
        std::size_t pixel_bytes     = std::size_t(batch) * width * height * 3;
        pixels                      = shared ? compute::DeviceBuffer(shared->pixels.data, pixel_bytes) : compute::DeviceBuffer(pixel_bytes);
        auxbytes                    = 0;
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            tensors[i].data = static_cast<std::byte*>(arena.data) + offsets[i];
        }
        for (auto& op : ops) {
            op.aux = reinterpret_cast<float*>(static_cast<std::byte*>(auxiliary.data) + auxbytes);
            auxbytes += op.aux_bytes;
        }
        allocated_bytes = arena.size + auxiliary.size + scratch.size + workspace.size + pixels.size;
        compute::check(cudaMemsetAsync(arena.data, 0, arena.size, stream));
    }
    NetworkPlan::~NetworkPlan() {
        cudaStreamSynchronize(stream);
        if (inference_exec) cudaGraphExecDestroy(inference_exec);
        if (inference_graph) cudaGraphDestroy(inference_graph);
        convolutions.clear();
        gemms.clear();
        handles.reset();
        cudaStreamDestroy(stream);
    }
    int NetworkPlan::append(Operation kind, std::string name, int input, int channels, int kernel, int skip, float probability, Affine affine) {
        compute::TensorView out = tensors[input];
        if (channels) out.c = channels;
        if (kind == Operation::convolution) {
            out.h /= kernel;
            out.w /= kernel;
        }
        if (kind == Operation::pool) {
            out.h = 1;
            out.w = 1;
        }
        out.scalar = (name == "stem.1" || name == "head.norm" || kind == Operation::dropout || (kind == Operation::residual && (tensors[skip].scalar == compute::Scalar::f32))) ? compute::Scalar::f32 : compute::Scalar::bf16;
        int index  = int(tensors.size());
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
    void NetworkPlan::matrix(std::string key, compute::TensorView aa, compute::TensorView bb, compute::TensorView out, bool ta, bool tb, const void* bias, float beta) {
        int ar = aa.n * aa.h * aa.w, br = bb.n * bb.h * bb.w, m = out.n * out.h * out.w, n = out.c, k = ta ? ar : aa.c;
        auto found = gemms.find(key);
        if (found == gemms.end()) {
            auto plan = std::make_unique<compute::MatrixPlan>();
            compute::check(cublasLtMatmulDescCreate(std::out_ptr(plan->operation), CUBLAS_COMPUTE_32F, CUDA_R_32F));
            cublasOperation_t transa = tb ? CUBLAS_OP_T : CUBLAS_OP_N, transb = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
            compute::check(cublasLtMatmulDescSetAttribute(plan->operation.get(), CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
            compute::check(cublasLtMatmulDescSetAttribute(plan->operation.get(), CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
            if (bias) {
                cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
                compute::check(cublasLtMatmulDescSetAttribute(plan->operation.get(), CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
                compute::check(cublasLtMatmulDescSetAttribute(plan->operation.get(), CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
            }
            compute::check(cublasLtMatrixLayoutCreate(std::out_ptr(plan->a), (bb.scalar == compute::Scalar::f32) ? CUDA_R_32F : CUDA_R_16BF, bb.c, br, bb.c));
            compute::check(cublasLtMatrixLayoutCreate(std::out_ptr(plan->b), (aa.scalar == compute::Scalar::f32) ? CUDA_R_32F : CUDA_R_16BF, aa.c, ar, aa.c));
            compute::check(cublasLtMatrixLayoutCreate(std::out_ptr(plan->output), (out.scalar == compute::Scalar::f32) ? CUDA_R_32F : CUDA_R_16BF, n, m, n));
            const auto cache = model.cache_directory / std::format("matrix-{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.json", m, n, k, ta, tb, (aa.scalar == compute::Scalar::f32), (bb.scalar == compute::Scalar::f32), (out.scalar == compute::Scalar::f32), bias != nullptr, workspace.size);
            compute::select_matrix(*plan, handles->blas.get(), stream, bb.data, aa.data, out.data, 1, beta, out.bytes(), workspace.data, workspace.size, cache);
            found = gemms.emplace(key, std::move(plan)).first;
        }
        const auto& p = *found->second;
        float alpha   = 1;
        compute::check(cublasLtMatmul(handles->blas.get(), p.operation.get(), &alpha, bb.data, p.a.get(), aa.data, p.b.get(), &beta, out.data, p.output.get(), out.data, p.output.get(), &p.algorithm, workspace.data, p.workspace_bytes, stream));
    }
    void NetworkPlan::convolution(std::string key, compute::TensorView x, Parameter& weight, compute::TensorView out, int kernel, int direction, const void* original_input) {
        auto found = convolutions.find(key);
        std::unordered_map<std::int64_t, void*> pointers{{1, x.data}, {2, weight.gpu.bf16}, {3, out.data}};
        if (direction == 0) pointers[4] = model.parameters.at(weight.name.substr(0, weight.name.size() - 6) + "bias").gpu.bf16;
        if (direction == 2) pointers[2] = const_cast<void*>(original_input);
        if (found == convolutions.end()) {
            auto graph = std::make_shared<cudnn_frontend::graph::Graph>();
            auto cache = model.cache_directory / "plans" / std::format("dnn_v2_{}_{}_{}_{}_{}_{}_{}.bin", batch, width, height, key, weight.shape[0], weight.shape[3], workspace.size);
            if (std::filesystem::exists(cache)) {
                const auto bytes = files::read_bytes(cache);
                dnn_check(graph->deserialize(handles->dnn.get(), bytes, false, false));
                ConvPlan plan;
                plan.graph     = graph;
                plan.workspace = graph->get_workspace_size();
                found          = convolutions.emplace(key, std::move(plan)).first;
                dnn_check(found->second.graph->execute(handles->dnn.get(), pointers, workspace.data));
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
            dnn_check(graph->build_operation_graph(handles->dnn.get()));
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
            files::write_bytes(cache, bytes);
            found = convolutions.emplace(key, std::move(plan)).first;
        }
        dnn_check(found->second.graph->execute(handles->dnn.get(), pointers, workspace.data));
    }
    void NetworkPlan::forward(bool stochastic) {
        for (auto& op : ops) {
            compute::TensorView x = tensors[op.input], out = tensors[op.output];
            ParameterView w = op.weight ? op.weight->gpu : ParameterView{}, b = op.bias ? op.bias->gpu : ParameterView{};
            switch (op.kind) {
            case Operation::convolution: convolution(op.name, x, *op.weight, out, op.kernel); break;
            case Operation::depthwise:
                {
                    if (x.scalar == compute::Scalar::f32) {
                        compute::TensorView bf16 = x;
                        bf16.scalar              = compute::Scalar::bf16;
                        bf16.data                = scratch.data;
                        convert(stream, bf16, x);
                        x = bf16;
                    }
                    depthwise(stream, out, x, w.bf16, b.bf16);
                    break;
                }
            case Operation::norm: layer_norm(stream, out, x, w.master, b.master, storage == ActivationStorage::retain ? op.aux : nullptr); break;
            case Operation::linear:
                {
                    if (x.scalar == compute::Scalar::f32) {
                        compute::TensorView bf16 = x;
                        bf16.scalar              = compute::Scalar::bf16;
                        bf16.data                = scratch.data;
                        convert(stream, bf16, x);
                        x = bf16;
                    }
                    matrix(op.name, x, {w.bf16, 1, 1, out.c, x.c, compute::Scalar::bf16}, out, false, true, b.bf16);
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
        compute::check(cudaStreamSynchronize(stream));
        compute::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        forward(false);
        compute::check(cudaStreamEndCapture(stream, &inference_graph));
        compute::check(cudaGraphInstantiate(&inference_exec, inference_graph, 0));
    }
    void NetworkPlan::infer(GpuRgb8 input, GpuResult result, cudaStream_t caller_stream) {
        preprocess(caller_stream, tensors[0], input.pixels, input.row_stride, std::size_t(input.height) * input.row_stride, nullptr, nullptr, false);
        compute::check(cudaGraphLaunch(inference_exec, caller_stream));
        softmax(caller_stream, tensors.back(), result.scores, result.indices);
    }
    TrainingPlan::TrainingPlan(Network& model, int n, int w, int h, TrainingPlan* shared) : NetworkPlan(model, n, w, h, ActivationStorage::retain, shared) {
        const auto placement = tensor_layout(tensors, ops, false);
        gradient_arena       = shared ? compute::DeviceBuffer(shared->gradient_arena.data, placement.bytes) : compute::DeviceBuffer(placement.bytes);
        labels               = shared ? compute::DeviceBuffer(shared->labels.data, n * 4) : compute::DeviceBuffer(n * 4);
        augment              = shared ? compute::DeviceBuffer(shared->augment.data, n * sizeof(Augment)) : compute::DeviceBuffer(n * sizeof(Augment));
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            auto g = tensors[i];
            g.data = static_cast<std::byte*>(gradient_arena.data) + placement.offsets[i];
            gradients.push_back(g);
        }
        UpdateState state;
        compute::check(cudaMemcpy(&state, model.optimizer->update, sizeof(state), cudaMemcpyDeviceToHost));
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
            compute::TensorView x = tensors[op.input], dy = gradients[op.output], dx = gradients[op.input];
            bool accumulate = ready[op.input];
            if (accumulate) dx.data = scratch.data;
            float* reduction = reinterpret_cast<float*>(static_cast<std::byte*>(scratch.data) + scratch.size - std::size_t(batch) * 129 * 3072 * 4 - 32 * 1024 * 1024);
            ParameterView w = op.weight ? op.weight->gpu : ParameterView{}, b = op.bias ? op.bias->gpu : ParameterView{};
            switch (op.kind) {
            case Operation::convolution:
                {
                    convolution(op.name + ".dx", dy, *op.weight, dx, op.kernel, 1);
                    compute::TensorView weight_grad{reduction, 1, 1, int(w.count) / x.c, x.c, compute::Scalar::bf16};
                    compute::TensorView input_shape = x;
                    input_shape.data                = weight_grad.data;
                    convolution(op.name + ".dw", dy, *op.weight, input_shape, op.kernel, 2, x.data);
                    compute::TensorView accumulated{w.grad, 1, 1, int(w.count) / x.c, x.c, compute::Scalar::f32};
                    add(stream, accumulated, accumulated, weight_grad);
                    bias_backward(stream, dy, b.grad);
                    break;
                }
            case Operation::depthwise: depthwise_backward(stream, dx, dy, x, w.bf16, w.grad, b.grad, reduction); break;
            case Operation::norm: layer_norm_backward(stream, dx, dy, x, w.master, op.aux, w.grad, b.grad, reduction); break;
            case Operation::linear:
                {
                    compute::TensorView input = x;
                    if (input.scalar == compute::Scalar::f32) {
                        input.data   = reduction;
                        input.scalar = compute::Scalar::bf16;
                        convert(stream, input, x);
                    }
                    matrix(op.name + ".dw", dy, input, {w.grad, 1, 1, dy.c, x.c, compute::Scalar::f32}, true, false, nullptr, 1);
                    compute::TensorView native_dx = dx;
                    if (dx.scalar == compute::Scalar::f32) {
                        native_dx.data   = reduction;
                        native_dx.scalar = compute::Scalar::bf16;
                    }
                    matrix(op.name + ".dx", dy, {w.bf16, 1, 1, dy.c, x.c, compute::Scalar::bf16}, native_dx, false, false);
                    if (dx.scalar == compute::Scalar::f32) convert(stream, dx, native_dx);
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
        compute::check(cudaMemsetAsync(labels.data, 0, labels.size, stream));
        backward(false);
        compute::check(cudaStreamSynchronize(stream));
        for (auto& [name, p] : model.parameters) compute::check(cudaMemsetAsync(p.gpu.grad, 0, p.gpu.count * 4, stream));
        for (bool frozen : std::array{true, false}) {
            cudaGraph_t& graph    = frozen ? frozen_graph : training_graph;
            cudaGraphExec_t& exec = frozen ? frozen_exec : training_exec;
            compute::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            preprocess(stream, tensors[0], static_cast<unsigned char*>(pixels.data), width * 3, std::size_t(width) * height * 3, static_cast<Augment*>(augment.data), model.optimizer->random, true);
            forward(!frozen);
            backward(frozen);
            random_advance(stream, model.optimizer->random);
            compute::check(cudaStreamEndCapture(stream, &graph));
            compute::check(cudaGraphInstantiate(&exec, graph, 0));
        }
    }
    void TrainingPlan::microbatch(const unsigned char* host_rgb, const int* host_labels, bool head_only, cudaStream_t caller_stream, cudaEvent_t copied) {
        compute::check(cudaMemcpyAsync(pixels.data, host_rgb, pixels.size, cudaMemcpyHostToDevice, caller_stream));
        compute::check(cudaMemcpyAsync(labels.data, host_labels, labels.size, cudaMemcpyHostToDevice, caller_stream));
        compute::check(cudaEventRecord(copied, caller_stream));
        compute::check(cudaGraphLaunch(head_only ? frozen_exec : training_exec, caller_stream));
    }
} // namespace genesia::convnext
