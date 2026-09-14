module;
#include "../../compute/tensor.h"
#include <cuda_fp16.h>
#include <genesia/cuda.h>

export module genesia.models.sdxl.unet;
import std;
import genesia.compute.inference;
import genesia.models.sdxl.workspace;
import genesia.models.sdxl.weights;

export namespace genesia::sdxl {
    struct Residual final {
        compute::Norm norm1;
        compute::Conv conv1;
        compute::Linear time;
        compute::Norm norm2;
        compute::Conv conv2;
        compute::Conv shortcut;

        Residual() = default;
        Residual(Checkpoint& source, const std::string& prefix, bool timed);
        void forward(compute::TensorView output, compute::TensorView input, compute::TensorView projected_time, const int* step, compute::InferenceRuntime& runtime, const Workspace& scratch, bool prepared_statistics = false) const;
    };
    struct Transformer final {
        compute::Norm norm1;
        compute::Linear qkv;
        compute::Linear self_output;
        compute::Norm norm2;
        compute::Linear query;
        compute::Linear kv;
        compute::Linear cross_output;
        compute::Norm norm3;
        compute::Linear expand;
        compute::Linear contract;

        Transformer(Checkpoint& source, const std::string& prefix);
        void forward(compute::TensorView values, compute::TensorView key, compute::TensorView value, const std::int32_t* lengths, compute::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
    struct SpatialTransformer final {
        compute::Norm norm;
        compute::Linear input;
        std::vector<Transformer> blocks;
        compute::Linear output;

        SpatialTransformer() = default;
        SpatialTransformer(Checkpoint& source, const std::string& prefix, int depth);
        void forward(compute::TensorView values, compute::TensorView sequence, std::span<const std::array<compute::TensorView, 2>> context, const std::int32_t* lengths, compute::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
    struct UNetStage final {
        std::vector<Residual> residuals;
        std::vector<SpatialTransformer> transformers;
        compute::Conv resize;
    };
    struct UNetState final {
        ::cuda::std::span<__half> current;
        ::cuda::std::span<__half> sequence;
        std::vector<::cuda::device_buffer<__half>> skips;
        std::vector<::cuda::device_buffer<__half>> time_storage;
        std::vector<compute::TensorView> time;
        std::vector<::cuda::device_buffer<__half>> context_storage;
        std::vector<std::array<compute::TensorView, 2>> context;
        ::cuda::device_buffer<std::int32_t> lengths;
        int height;
        int width;

        UNetState(::cuda::stream_ref stream, int height, int width);
    };
    struct UNet final {
        compute::Linear time_input;
        compute::Linear time_output;
        compute::Linear label_input;
        compute::Linear label_output;
        compute::Conv input;
        std::array<UNetStage, 3> down;
        Residual middle_input;
        SpatialTransformer middle;
        Residual middle_output;
        std::array<UNetStage, 3> up;
        compute::Norm norm;
        compute::Conv output;

        explicit UNet(Checkpoint& source);
        void prepare(UNetState& state, compute::TensorView context, compute::TensorView condition, const float* times, int steps, compute::InferenceRuntime& runtime, const Workspace& scratch) const;
        void forward(compute::TensorView output, compute::TensorView input, const int* step, UNetState& state, compute::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
} // namespace genesia::sdxl
