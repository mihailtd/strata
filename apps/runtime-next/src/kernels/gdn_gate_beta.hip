// Real HIP kernel: GatedDeltaNet's per-head decay/beta computation
// (`beta = b.sigmoid()`, `g = -self.A_log.float().exp() * F.softplus(a.float()
// + self.dt_bias)` in Qwen3_5GatedDeltaNet.forward). `a`/`b` are the raw
// per-token, per-head projections (`in_proj_a`/`in_proj_b`); `A_log`/
// `dt_bias` are real trained per-layer parameters, shape [num_v_heads]
// each, same as everything else here. Tiny (num_v_heads=32 elements for
// this model) but kept as its own kernel rather than a host round-trip, to
// avoid a mid-layer GPU<->host sync in the real decode loop.
//
// softplus(x) = log(1+exp(x)), using the same numerically-stable threshold
// PyTorch's own F.softplus default uses (beta=1, threshold=20): for x>20,
// softplus(x) is indistinguishable from x in fp32, so return x directly
// rather than risking overflow in exp(x).

#include "bf16_utils.hip"

__device__ __forceinline__ float softplus_f32(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

extern "C" void launch_gdn_gate_beta_batched_bf16(
    const void* a, const void* b, const float* a_log, const float* dt_bias,
    float* g_out, float* beta_out, int num_heads, int batch,
    long long a_stride, long long out_stride, void* stream);

extern "C" __global__ void gdn_gate_beta_bf16_kernel(
    const unsigned short* a,
    const unsigned short* b,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* g_out,
    float* beta_out,
    int num_heads,
    long long a_stride,   // elements between sequences in a/b (in_proj row width)
    long long out_stride  // elements between sequences in g_out/beta_out
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;
    // blockIdx.y is the sequence; a_log/dt_bias are WEIGHTS, shared by all.
    const int seq = blockIdx.y;
    a = a + (size_t)seq * (size_t)a_stride;
    b = b + (size_t)seq * (size_t)a_stride;
    g_out = g_out + (size_t)seq * (size_t)out_stride;
    beta_out = beta_out + (size_t)seq * (size_t)out_stride;
    float av = bf16_bits_to_f32(a[h]);
    float bv = bf16_bits_to_f32(b[h]);
    g_out[h] = -expf(a_log[h]) * softplus_f32(av + dt_bias[h]);
    beta_out[h] = 1.0f / (1.0f + expf(-bv));
}

// `stream` (§93): see `rmsnorm.hip`'s launcher comment.
extern "C" void launch_gdn_gate_beta_bf16(
    const void* a,
    const void* b,
    const float* a_log,
    const float* dt_bias,
    float* g_out,
    float* beta_out,
    int num_heads,
    void* stream
) {
    launch_gdn_gate_beta_batched_bf16(a, b, a_log, dt_bias, g_out, beta_out, num_heads, 1, 0, 0, stream);
}

// Batched: one launch for `batch` sequences.
extern "C" void launch_gdn_gate_beta_batched_bf16(
    const void* a,
    const void* b,
    const float* a_log,
    const float* dt_bias,
    float* g_out,
    float* beta_out,
    int num_heads,
    int batch,
    long long a_stride,
    long long out_stride,
    void* stream
) {
    int threads = num_heads < 256 ? num_heads : 256;
    int blocks = (num_heads + threads - 1) / threads;
    hipLaunchKernelGGL(
        gdn_gate_beta_bf16_kernel,
        dim3(blocks, batch), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)a, (const unsigned short*)b, a_log, dt_bias, g_out, beta_out, num_heads,
        a_stride, out_stride
    );
}
