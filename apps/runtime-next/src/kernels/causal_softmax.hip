// Real HIP kernel: §99 (batched-prefill causal attention). Row-wise
// numerically-stable softmax over a REAL PRECOMPUTED score matrix
// (`S = Q @ K^T * scaling`, computed by `blas::gemm_qkt_bf16`), with a
// CAUSAL mask applied per row -- the piece that turns a plain GEMM-based
// QK^T into real causal attention.
//
// Why a separate kernel instead of folding masking into the GEMM itself:
// `gemm_qkt_bf16` computes the FULL `[T, kv_len]` score matrix in one
// batched call (every query row against every key, including keys that
// are causally in the FUTURE relative to that query row) -- real, correct
// GEMM math, just not yet causally valid on its own. Masking has to
// happen AFTER the GEMM, before softmax: for query row `t` (0-indexed
// within this prefill chunk), the real valid key range is
// `[0, start_position + t]` inclusive -- columns beyond that must
// contribute exactly zero probability, matching this model's real
// `eager_attention_forward`'s own causal-mask-before-softmax semantics
// (see `attention.hip`'s header for the single-query decode case this
// generalizes).
//
// Masking is applied by SKIPPING invalid columns in the max/sum
// reductions and writing an explicit 0.0 into the output for them --
// mathematically identical to masking to -inf before softmax (which
// would also produce exactly 0 probability there), but avoids a separate
// masked-write pass. Zeroed probabilities then make `gemm_pv_bf16`'s
// `O = P @ V` naturally sum only over the real causal range, with no
// special-casing needed in that GEMM at all.
//
// One block per query row (`blockIdx.x`), in-place on `scores`. Same
// numerically-stable max-then-exp-then-normalize idiom, and the same
// block-level power-of-two-halving reduction, `attention.hip` and
// `rmsnorm.hip` already established.

#include "bf16_utils.hip"

extern "C" __global__ void causal_softmax_bf16_kernel(
    unsigned short* __restrict__ scores,   // [num_rows, kv_len], in-place
    int kv_len,
    const int* __restrict__ start_position_ptr,  // device scalar
    int num_rows
) {
    extern __shared__ float reduce_scratch[]; // [blockDim.x]

    const int row = blockIdx.x;
    if (row >= num_rows) return;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Real causal boundary for this row: query at absolute position
    // (start_position + row) may attend to keys [0, start_position + row].
    const int valid_len = *start_position_ptr + row + 1;

    unsigned short* score_row = scores + (size_t)row * (size_t)kv_len;

    // ---- Phase 1: max over the VALID range only ----
    float local_max = -3.402823466e38f; // -FLT_MAX
    for (int j = tid; j < valid_len; j += stride) {
        local_max = fmaxf(local_max, bf16_bits_to_f32(score_row[j]));
    }
    reduce_scratch[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduce_scratch[tid] = fmaxf(reduce_scratch[tid], reduce_scratch[tid + s]);
        }
        __syncthreads();
    }
    const float max_score = reduce_scratch[0];
    __syncthreads();

    // ---- Phase 2: exp(score - max) for valid columns, 0 for masked ones,
    // plus a block-level sum reduction over the valid range. ----
    float local_sum = 0.0f;
    for (int j = tid; j < kv_len; j += stride) {
        if (j < valid_len) {
            float e = expf(bf16_bits_to_f32(score_row[j]) - max_score);
            score_row[j] = f32_to_bf16_bits(e); // unnormalized, normalized in phase 3
            local_sum += e;
        } else {
            score_row[j] = f32_to_bf16_bits(0.0f); // causally masked -> zero probability
        }
    }
    reduce_scratch[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduce_scratch[tid] += reduce_scratch[tid + s];
        }
        __syncthreads();
    }
    const float sum_exp = reduce_scratch[0];
    __syncthreads();

    // ---- Phase 3: normalize the valid range (masked columns are already
    // exactly 0.0 and stay that way). ----
    for (int j = tid; j < valid_len; j += stride) {
        float e = bf16_bits_to_f32(score_row[j]);
        score_row[j] = f32_to_bf16_bits(e / sum_exp);
    }
}

// Plain C entry point. `threads` must be a power of two; shared memory
// sized for `threads` floats (the reduction scratch only -- unlike
// `attention.hip`, there is no per-row Q/score caching here, since scores
// are already real, precomputed GEMM output, not derived inside this
// kernel). `start_position_ptr` is a DEVICE pointer -- same
// graph-capture-safety convention every other position-dependent launcher
// in this crate already follows (see `rope.hip`'s own comment).
extern "C" void launch_causal_softmax_bf16(
    void* scores,
    int kv_len,
    const int* start_position_ptr,
    int num_rows,
    int threads,
    void* stream
) {
    size_t shmem_bytes = (size_t)threads * sizeof(float);
    hipLaunchKernelGGL(
        causal_softmax_bf16_kernel,
        dim3(num_rows), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (unsigned short*)scores, kv_len, start_position_ptr, num_rows
    );
}
