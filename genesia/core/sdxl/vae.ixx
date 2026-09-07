module;
#include <genesia/cuda.h>

export module genesia.sdxl.vae;
import std;
import genesia.neural.inference_runtime;
import genesia.sdxl.workspace;
import genesia.sdxl.weights;
import genesia.sdxl.unet;

export namespace genesia::sdxl {
    struct VaeStage final {
        std::vector<Residual> blocks;
        neural::Conv resize;
    };
    struct VAE final {
        neural::Conv post_quant;
        neural::Conv input;
        Residual middle_input;
        neural::Norm attention_norm;
        neural::Linear qkv;
        neural::Conv projection;
        Residual middle_output;
        std::array<VaeStage, 4> up;
        neural::Norm norm;
        neural::Conv output;

        explicit VAE(Checkpoint& source);
        void forward(neural::TensorView output, neural::TensorView latent, neural::TensorView current, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
} // namespace genesia::sdxl
