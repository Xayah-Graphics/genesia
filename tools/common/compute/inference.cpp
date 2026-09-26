module;
#include "inference-kernels.h"
#include "tensor.h"
#include <cublasLt.h>
// cuDNN plan JSON uses its own float encoding, separate from PNG and CLI metadata.
#define nlohmann cudnn_json
#include <cudnn_frontend.h>
#undef nlohmann
#include <genesia/cuda.h>

module tools.compute.inference;

import tools.files;
import std;

namespace tools::compute {
    struct InferenceRuntime::ConvPlan final {
        std::array<int, 8> key;
        cudnn_frontend::graph::Graph graph;

        explicit ConvPlan(const std::array<int, 8>& key);
    };

    struct InferenceRuntime::AttentionPlan final {
        std::array<int, 7> key;
        cudnn_frontend::graph::Graph graph;

        explicit AttentionPlan(const std::array<int, 7>& shape);
    };

    void check(cudnn_frontend::error_t status) {
        if (status.is_bad()) throw std::runtime_error{status.get_message()};
    }

    MatmulPlan::MatmulPlan(const MatmulShape& shape) : shape{shape} {
        const auto [m, n, k, bias] = shape;
        check(cublasLtMatmulDescCreate(std::out_ptr(operation), CUBLAS_COMPUTE_32F, CUDA_R_32F));
        constexpr cublasOperation_t transpose = CUBLAS_OP_T;
        check(cublasLtMatmulDescSetAttribute(operation.get(), CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
        const cublasLtEpilogue_t epilogue = bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
        check(cublasLtMatmulDescSetAttribute(operation.get(), CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        check(cublasLtMatrixLayoutCreate(std::out_ptr(a), CUDA_R_16F, k, n, k));
        check(cublasLtMatrixLayoutCreate(std::out_ptr(b), CUDA_R_16F, k, m, k));
        check(cublasLtMatrixLayoutCreate(std::out_ptr(output), CUDA_R_16F, n, m, n));
    }

    InferenceRuntime::ConvPlan::ConvPlan(const std::array<int, 8>& shape) : key{shape} {
        const auto [n, h, w, input_width, output_width, kernel, stride, padding] = shape;
        graph.set_io_data_type(cudnn_frontend::DataType_t::HALF).set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT).set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
        auto input               = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("input").set_uid(1).set_dim({n, input_width, h, w}).set_stride({static_cast<std::int64_t>(h) * w * input_width, 1, static_cast<std::int64_t>(w) * input_width, input_width}));
        auto weight              = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("weight").set_uid(2).set_dim({output_width, input_width, kernel, kernel}).set_stride({static_cast<std::int64_t>(kernel) * kernel * input_width, 1, static_cast<std::int64_t>(kernel) * input_width, input_width}));
        auto result              = graph.conv_fprop(input, weight, cudnn_frontend::graph::Conv_fprop_attributes{}.set_padding({padding, padding}).set_stride({stride, stride}).set_dilation({1, 1}));
        auto bias                = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("bias").set_uid(3).set_dim({1, output_width, 1, 1}).set_stride({output_width, 1, output_width, output_width}));
        result                   = graph.pointwise(result, bias, cudnn_frontend::graph::Pointwise_attributes{}.set_mode(cudnn_frontend::PointwiseMode_t::ADD));
        const int output_height  = (h + 2 * padding - kernel) / stride + 1;
        const int output_columns = (w + 2 * padding - kernel) / stride + 1;
        result->set_output(true).set_uid(4).set_dim({n, output_width, output_height, output_columns}).set_stride({static_cast<std::int64_t>(output_height) * output_columns * output_width, 1, static_cast<std::int64_t>(output_columns) * output_width, output_width});
        check(graph.validate());
    }

    InferenceRuntime::AttentionPlan::AttentionPlan(const std::array<int, 7>& shape) : key{shape} {
        const auto [batch, queries, keys, heads, dimension, query_stride, key_stride] = shape;
        graph.set_io_data_type(cudnn_frontend::DataType_t::HALF).set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT).set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
        auto q          = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("q").set_uid(1).set_dim({batch, heads, queries, dimension}).set_stride({static_cast<std::int64_t>(queries) * query_stride, dimension, query_stride, 1}));
        auto k          = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("k").set_uid(2).set_dim({batch, heads, keys, dimension}).set_stride({static_cast<std::int64_t>(keys) * key_stride, dimension, key_stride, 1}));
        auto v          = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("v").set_uid(3).set_dim({batch, heads, keys, dimension}).set_stride({static_cast<std::int64_t>(keys) * key_stride, dimension, key_stride, 1}));
        auto attributes = cudnn_frontend::graph::SDPA_attributes{}.set_generate_stats(false).set_attn_scale(1.0F / std::sqrt(static_cast<float>(dimension)));
        attributes.set_bias(graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_uid(7).set_dim({batch, heads, queries, keys}).set_stride({std::int64_t(heads) * queries * keys, std::int64_t(queries) * keys, keys, 1})));
        auto [o, stats] = graph.sdpa(q, k, v, attributes);
        o->set_output(true).set_uid(4).set_dim({batch, heads, queries, dimension}).set_stride({static_cast<std::int64_t>(queries) * heads * dimension, dimension, heads * dimension, 1});
        check(graph.validate());
    }

    InferenceRuntime::InferenceRuntime(const ::cuda::stream_ref execution, const std::filesystem::path& directory) : stream{execution}, handles{execution.get()}, workspace{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz << 30, ::cuda::no_init}, cache_directory{plan_directory(directory, stream.device().get())}, workspace_data{workspace.data()}, workspace_bytes{workspace.size()} {}

    InferenceRuntime::~InferenceRuntime() {
        stream.sync();
    }

    ::cuda::device_buffer<std::byte> InferenceRuntime::finish_preparation() {
        stream.sync();
        workspace       = ::cuda::device_buffer<std::byte>{stream, ::cuda::device_default_memory_pool(stream.device()), required_workspace, ::cuda::no_init};
        workspace_data  = workspace.data();
        workspace_bytes = workspace.size();
        return std::move(workspace);
    }

    void InferenceRuntime::linear(const TensorView result, const TensorView input, const Linear& layer) {
        const MatmulShape shape{input.n * input.h * input.w, layer.weight.w, input.c, layer.bias.data != nullptr};
        auto plan          = std::ranges::find_if(matmuls, [&](const MatmulPlan& value) { return value.shape == shape; });
        const bool prepare = plan == matmuls.end();
        if (prepare) plan = matmuls.emplace(matmuls.end(), shape);
        if (layer.bias.data) check(cublasLtMatmulDescSetAttribute(plan->operation.get(), CUBLASLT_MATMUL_DESC_BIAS_POINTER, &layer.bias.data, sizeof(layer.bias.data)));
        constexpr float alpha = 1.0F, beta = 0.0F;
        if (prepare) {
            const auto file = cache_directory / std::format("gemm-{}-{}-{}-{}.json", shape.rows, shape.columns, shape.reduction, shape.bias);
            select_matrix(*plan, handles.blas.get(), stream.get(), layer.weight.data, input.data, result.data, alpha, beta, result.bytes(), workspace_data, 1uz << 30, file);
            required_workspace = std::max(required_workspace, plan->workspace_bytes);
        }
        check(cublasLtMatmul(handles.blas.get(), plan->operation.get(), &alpha, layer.weight.data, plan->a.get(), input.data, plan->b.get(), &beta, result.data, plan->output.get(), result.data, plan->output.get(), &plan->algorithm, workspace_data, workspace_bytes, stream.get()));
    }

    void InferenceRuntime::convolution(const TensorView output, const TensorView input, const Conv& layer) {
        const std::array<int, 8> key{input.n, input.h, input.w, input.c, layer.weight.n, layer.weight.h, layer.stride, layer.padding};
        auto plan = std::ranges::find_if(convolutions, [&](const ConvPlan& value) { return value.key == key; });
        std::unordered_map<std::int64_t, void*> tensors{{1, input.data}, {2, layer.weight.data}, {3, layer.bias.data}, {4, output.data}};
        if (plan == convolutions.end()) {
            plan            = convolutions.emplace(convolutions.end(), key);
            const auto file = cache_directory / ("conv-" + cudnn_json::json(key).dump() + ".bin");
            if (std::filesystem::exists(file)) check(plan->graph.deserialize(handles.dnn.get(), files::read_bytes(file), false, false));
            else {
                check(plan->graph.build_operation_graph(handles.dnn.get()));
                check(plan->graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}));
                plan->graph.deselect_numeric_notes({cudnn_frontend::NumericalNote_t::NONDETERMINISTIC, cudnn_frontend::NumericalNote_t::REDUCED_PRECISION_REDUCTION});
                plan->graph.deselect_workspace_greater_than((1uz << 30));
                check(plan->graph.check_support(handles.dnn.get()));
                check(plan->graph.build_plans(cudnn_frontend::BuildPlanPolicy_t::ALL));
                ::cuda::device_buffer<std::byte> temporary{stream, ::cuda::device_default_memory_pool(stream.device()), output.bytes(), ::cuda::no_init};
                tensors[4] = temporary.data();
                check(plan->graph.autotune(handles.dnn.get(), tensors, workspace_data));
                tensors[4] = output.data;
                std::vector<std::uint8_t> bytes;
                check(plan->graph.serialize(bytes, false));
                files::write_bytes(file, bytes);
            }
            required_workspace = std::max(required_workspace, static_cast<std::size_t>(plan->graph.get_workspace_size()));
        }
        check(plan->graph.execute(handles.dnn.get(), tensors, workspace_data));
    }

    void InferenceRuntime::layer_norm(const TensorView output, const TensorView input, const Norm& layer) {
        kernels::layer_norm(stream, output.data, input.data, layer.weight.data, layer.bias.data, input.n * input.h * input.w, input.c, layer.epsilon);
    }

    void InferenceRuntime::attention(const TensorView output, const TensorView query, const TensorView key, const TensorView value, const int heads, const int query_stride, const int key_stride, const TensorView bias) {
        const int dimension = output.c / heads;
        const int queries   = query.h * query.w;
        const int keys      = key.h * key.w;
        const std::array<int, 7> shape{query.n, queries, keys, heads, dimension, query_stride, key_stride};
        auto plan = std::ranges::find_if(attentions, [&](const AttentionPlan& item) { return item.key == shape; });
        std::unordered_map<std::int64_t, void*> tensors{{1, query.data}, {2, key.data}, {3, value.data}, {4, output.data}, {7, bias.data}};
        if (plan == attentions.end()) {
            plan            = attentions.emplace(attentions.end(), shape);
            const auto file = cache_directory / ("attention-" + cudnn_json::json(shape).dump() + ".bin");
            if (std::filesystem::exists(file)) check(plan->graph.deserialize(handles.dnn.get(), files::read_bytes(file), false, false));
            else {
                check(plan->graph.build_operation_graph(handles.dnn.get()));
                check(plan->graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}));
                plan->graph.deselect_numeric_notes({cudnn_frontend::NumericalNote_t::NONDETERMINISTIC, cudnn_frontend::NumericalNote_t::REDUCED_PRECISION_REDUCTION});
                plan->graph.deselect_workspace_greater_than((1uz << 30));
                check(plan->graph.check_support(handles.dnn.get()));
                check(plan->graph.build_plans(cudnn_frontend::BuildPlanPolicy_t::ALL));
                check(plan->graph.autotune(handles.dnn.get(), tensors, workspace_data));
                std::vector<std::uint8_t> bytes;
                check(plan->graph.serialize(bytes, false));
                files::write_bytes(file, bytes);
            }
            required_workspace = std::max(required_workspace, static_cast<std::size_t>(plan->graph.get_workspace_size()));
        }
        check(plan->graph.execute(handles.dnn.get(), tensors, workspace_data));
    }

} // namespace tools::compute
