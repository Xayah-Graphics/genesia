module;
#include "../../compute/tensor.h"
#include <cuda_fp16.h>
#include <genesia/cuda.h>

export module genesia.models.sdxl.text;
import std;
import genesia.compute.inference;
import genesia.models.sdxl.workspace;
import genesia.models.sdxl.weights;
import genesia.models.sdxl.tokenizer;

export namespace genesia::sdxl {
    struct ClipBlock final {
        compute::Norm norm1;
        compute::Linear qkv;
        compute::Linear projection;
        compute::Norm norm2;
        compute::Linear expand;
        compute::Linear contract;
    };
    struct Encoding final {
        ::cuda::device_buffer<__half> sequence;
        ::cuda::device_buffer<__half> pooled;
    };
    struct Clip final {
        int width;
        bool large;
        compute::TensorView token;
        compute::TensorView position;
        std::vector<ClipBlock> blocks;
        compute::Norm final_norm;
        compute::Linear projection;
        ::cuda::device_buffer<__half> empty;

        Clip(Checkpoint& checkpoint, bool large);
        void forward(compute::TensorView sequence, const std::int32_t* ids, compute::InferenceRuntime& runtime, const Workspace& scratch) const;
        void pool(compute::TensorView pooled, compute::TensorView sequence, const std::int32_t* eos, compute::InferenceRuntime& runtime, const Workspace& scratch) const;
        void prepare_empty(compute::InferenceRuntime& runtime, const Workspace& scratch);
        Encoding encode(const Tokens& positive, const Tokens& negative, compute::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
} // namespace genesia::sdxl
