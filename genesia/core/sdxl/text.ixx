module;
#include <cuda_fp16.h>
#include <genesia/cuda.h>

export module genesia.sdxl.text;
import std;
import genesia.neural.inference_runtime;
import genesia.sdxl.workspace;
import genesia.sdxl.weights;
import genesia.sdxl.tokenizer;

export namespace genesia::sdxl {
    struct ClipBlock final {
        neural::Norm norm1;
        neural::Linear qkv;
        neural::Linear projection;
        neural::Norm norm2;
        neural::Linear expand;
        neural::Linear contract;
    };
    struct Encoding final {
        ::cuda::device_buffer<__half> sequence;
        ::cuda::device_buffer<__half> pooled;
    };
    struct Clip final {
        int width;
        bool large;
        neural::TensorView token;
        neural::TensorView position;
        std::vector<ClipBlock> blocks;
        neural::Norm final_norm;
        neural::Linear projection;
        ::cuda::device_buffer<__half> empty;

        Clip(Checkpoint& checkpoint, bool large);
        void forward(neural::TensorView sequence, const std::int32_t* ids, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
        void pool(neural::TensorView pooled, neural::TensorView sequence, const std::int32_t* eos, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
        void prepare_empty(neural::InferenceRuntime& runtime, const Workspace& scratch);
        Encoding encode(const Tokens& positive, const Tokens& negative, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
} // namespace genesia::sdxl
