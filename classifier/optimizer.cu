#include "kernel-common.cuh"
namespace classifier {
    __global__ void norm_kernel(ParameterView p, UpdateState* state) {
        if (state->step <= state->head_only_steps && !p.head) return;
        __shared__ cub::BlockReduce<float, 256>::TempStorage shared;
        float sum = 0;
        for (std::size_t i = blockIdx.x * 256 + threadIdx.x; i < p.count; i += gridDim.x * 256) sum = fmaf(p.grad[i], p.grad[i], sum);
        sum = cub::BlockReduce<float, 256>(shared).Sum(sum);
        if (threadIdx.x == 0) atomicAdd(&state->norm2, sum);
    }
    __global__ void begin_kernel(UpdateState* s) {
        ++s->step;
        s->norm2 = 0;
        s->loss  = 0;
        s->clip  = 1;
    }
    __global__ void clip_kernel(UpdateState* s) {
        s->clip = fminf(1.f, s->clip_norm / (sqrtf(s->norm2) + 1e-6f));
    }
    __global__ void adam_kernel(ParameterView p, UpdateState* s) {
        std::size_t i = std::size_t(blockIdx.x) * 256 + threadIdx.x;
        if (i >= p.count) return;
        if (s->step <= s->head_only_steps && !p.head) {
            p.grad[i] = 0;
            return;
        }
        int age        = p.head ? s->step : s->step - s->head_only_steps;
        float progress = fminf(fmaxf(float(s->step - s->head_only_steps - 1) / max(s->schedule_steps - s->head_only_steps - 1, 1), 0.f), 1.f);
        float factor   = s->step > s->schedule_steps ? .05f : fminf(fmaxf(float(s->step - s->head_only_steps) / s->warmup_steps, 0.f), 1.f) * (.05f + .95f * (1.f + cosf(3.14159265358979323846f * progress)) * .5f);
        float lr       = s->step <= s->head_only_steps ? s->head_only_lr : (p.head ? s->head_lr : s->backbone_lr) * factor;
        float grad = p.grad[i] * s->clip, m = .9f * p.m[i] + .1f * grad, v = .999f * p.v[i] + .001f * grad * grad;
        p.m[i]       = m;
        p.v[i]       = v;
        float weight = p.master[i] * (1.f - (p.decay ? lr * s->weight_decay : 0.f));
        weight -= lr * (m / (1.f - powf(.9f, float(age)))) / (sqrtf(v / (1.f - powf(.999f, float(age)))) + 1e-8f);
        p.master[i]                            = weight;
        static_cast<__nv_bfloat16*>(p.bf16)[i] = __float2bfloat16_rn(weight);
        p.grad[i]                              = 0;
    }
    void gradient_norm(cudaStream_t s, ParameterView p, UpdateState* state) {
        norm_kernel<<<128, 256, 0, s>>>(p, state);
    }
    void optimizer_begin(cudaStream_t s, UpdateState* state) {
        begin_kernel<<<1, 1, 0, s>>>(state);
    }
    void optimizer_clip(cudaStream_t s, UpdateState* state) {
        clip_kernel<<<1, 1, 0, s>>>(state);
    }
    void adamw(cudaStream_t s, ParameterView p, UpdateState* state) {
        adam_kernel<<<(p.count + 255) / 256, 256, 0, s>>>(p, state);
    }
} // namespace classifier
