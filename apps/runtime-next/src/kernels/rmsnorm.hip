// Real HIP kernel: RMSNorm, matching apps/runtime-ipwf/fused_norm.py's
// ExactRMSNorm exactly (unit_offset=True case -- what Qwen3_5RMSNorm uses,
// as opposed to Qwen3_5RMSNormGated's different weight/gate order):
//
//   normed = x.float() * rsqrt(mean(x.float()^2, dim=-1) + eps)
//   out    = (normed * (1.0 + weight.float())).to(input_dtype)
//
// All internal math in fp32 regardless of input/output dtype, matching that
// Python implementation's own docstring reasoning, not a simplification
// made here.
//
// bf16 in this file is carried as `unsigned short` (no native HIP bf16
// scalar type is used, to keep the ABI a plain C `unsigned short*` on the
// Rust side, matching model_loader.rs's own bf16-as-raw-bytes convention).
// bf16 -> f32 is the top 16 bits of an f32, the same conversion already
// verified correct against an independent Python computation in
// model_loader.rs. f32 -> bf16 rounds to nearest-even, matching PyTorch's
// own bf16 rounding (not truncation, which would be a different, less
// accurate operation than what the Python reference actually does).

#include "bf16_utils.hip"

extern "C" __global__ void rmsnorm_bf16_kernel(
    const unsigned short* __restrict__ x,
    const unsigned short* __restrict__ weight,
    unsigned short* __restrict__ out,
    int hidden_size,
    float eps
) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const unsigned short* xrow = x + (size_t)row * (size_t)hidden_size;
    unsigned short* orow = out + (size_t)row * (size_t)hidden_size;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float sumsq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = bf16_bits_to_f32(xrow[i]);
        sumsq += v * v;
    }
    sdata[tid] = sumsq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    const float mean_sq = sdata[0] / (float)hidden_size;
    const float rms = rsqrtf(mean_sq + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        float xv = bf16_bits_to_f32(xrow[i]);
        float wv = bf16_bits_to_f32(weight[i]);
        float normed = xv * rms;
        float outv = normed * (1.0f + wv);
        orow[i] = f32_to_bf16_bits(outv);
    }
}

// Plain C entry point -- the only symbol Rust links against. `num_rows`
// independent normalizations (one block each), `hidden_size` elements per
// row, `threads` threads per block (caller picks; must be a power of two
// for the reduction loop above).
// `stream` (§93): NULL reproduces the original default-stream behavior
// exactly; a real explicitly-created stream is required for this launch to
// be capturable into a HIP Graph (the null/legacy stream cannot be used at
// all while another stream is under capture).
extern "C" void launch_rmsnorm_bf16(
    const void* x,
    const void* weight,
    void* out,
    int num_rows,
    int hidden_size,
    float eps,
    int threads,
    void* stream
) {
    size_t shmem_bytes = (size_t)threads * sizeof(float);
    hipLaunchKernelGGL(
        rmsnorm_bf16_kernel,
        dim3(num_rows), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (const unsigned short*)x, (const unsigned short*)weight, (unsigned short*)out,
        hidden_size, eps
    );
}
