module;
#include "../../compute/tensor.h"
#include <genesia/cuda.h>

export module genesia.models.sdxl.vae;
import std;
import genesia.compute.inference;
import genesia.models.sdxl.workspace;
import genesia.models.sdxl.weights;
import genesia.models.sdxl.unet;

export namespace genesia::sdxl {
    struct VaeStage final {
        std::vector<Residual> blocks;
        compute::Conv resize;
    };
    struct VaeAttention final {
        compute::Norm norm;
        compute::Linear qkv;
        compute::Conv projection;

        VaeAttention(Checkpoint& source, const std::string& prefix);
        void forward(compute::TensorView current, compute::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
    struct VAEEncoder final {
        compute::Conv input;
        std::array<VaeStage, 4> down;
        Residual middle_input;
        VaeAttention attention;
        Residual middle_output;
        compute::Norm norm;
        compute::Conv output;
        compute::Conv quant;

        explicit VAEEncoder(Checkpoint& source);
        void forward(compute::TensorView latent, compute::TensorView pixels, compute::TensorView current, compute::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
    struct VAE final {
        compute::Conv post_quant;
        compute::Conv input;
        Residual middle_input;
        VaeAttention attention;
        Residual middle_output;
        std::array<VaeStage, 4> up;
        compute::Norm norm;
        compute::Conv output;

        explicit VAE(Checkpoint& source);
        void forward(compute::TensorView output, compute::TensorView latent, compute::TensorView current, compute::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
} // namespace genesia::sdxl
