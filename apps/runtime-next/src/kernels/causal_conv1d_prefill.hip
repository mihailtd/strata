// Real HIP kernel: §103 (real batched-prefill causal conv1d, eliminating
// the per-token `causal_conv1d_update` loop `gdn_layer_forward_prefill`
// used until now). Motivated by a direct comparison against
// `apps/runtime-llama/llama.cpp/src/models/qwen35.cpp`'s own real GDN
// prefill path (`ggml_ssm_conv`): llama.cpp issues ONE batched conv
// launch for the whole chunk; this crate's own per-token loop issued
// `num_tokens` separate kernel launches per layer (1,296 total across 24
// GDN layers for a real 54-token prompt) purely from real dispatch
// overhead, not real compute.
//
// UNLIKE GatedDeltaNet's own delta-rule recurrence (§100's real reason
// for a genuinely chunked UT-transform algorithm), causal conv1d has a
// BOUNDED lookback (`kernel_size=4`, i.e. only the current token and the
// 3 before it ever matter) -- there is no unbounded-lookback state to
// approximate or decompose. Batching this over a whole prefill chunk is
// therefore an EXACT parallel restatement of the same per-token formula,
// not an approximation: one thread owns one channel `c` for the WHOLE
// chunk (channels are fully independent, depthwise/groups=conv_dim,
// exactly like the per-token kernel already established), and walks
// `num_tokens` positions sequentially WITHIN that single thread, carrying
// a small `kernel_size`-wide register window forward -- the real
// sequential dependency (`catted = [prior 3 taps, current]`) is honored
// exactly, just without a host-side loop and `num_tokens` separate kernel
// launches to do it.
//
// Formula, per (token t, channel c), identical to `causal_conv1d_update`'s
// own real per-token formula:
//   catted   = concat(window[0..state_len), hidden_states[t,c])
//   raw      = sum_k catted[k] * weight[c,k]
//   out[t,c] = silu(raw)
//   window   = catted[1..kernel_size)   // slides forward one token
// After the last token, `window` (the real trailing `kernel_size-1` taps)
// is written back into `conv_state` for the NEXT call (another prefill
// chunk, or single-token decode) to continue from -- bit-for-bit the same
// real continuation contract `causal_conv1d_update`'s own in-place state
// update already provides.
//
// `src` is read as a STRIDED view (`src_stride`/`src_offset`), not
// necessarily tightly packed -- lets the caller read this layer's own
// `[T, GDN_CONV_DIM]` sub-range directly out of the wider
// `[T, GDN_IN_PROJ_COMBINED_DIM]` in-proj GEMM output, with zero physical
// extraction (the real "zero-copy strided view" technique
// `qwen35.cpp`'s own `ggml_view_4d` calls use, applied here via this
// kernel's own read-side stride instead of a separate `extract_range`
// pass). `out` stays tightly packed `[T, conv_dim]` -- every downstream
// consumer (`l2norm_bf16`, `gdn_chunk_broadcast_scale_bf16`, and
// GDN's real value/query/key split) already expects that exact layout.

#include "bf16_utils.hip"

// `kernel_size` is real-model-fixed at 4 (`state_len=3`); 8 is a safe
// register-array upper bound with a runtime `kernel_size` still driving
// every real loop bound below (no silent truncation for a hypothetical
// larger kernel -- would simply fail the `assert` in the launcher).
#define CAUSAL_CONV1D_PREFILL_MAX_KERNEL_SIZE 8

extern "C" __global__ void causal_conv1d_prefill_bf16_kernel(
    const unsigned short* __restrict__ src,    // [num_tokens, src_stride], read at [.., src_offset+c]
    int src_stride,
    int src_offset,
    unsigned short* __restrict__ conv_state,    // [conv_dim, state_len], read+written in place
    const unsigned short* __restrict__ weight,   // [conv_dim, kernel_size]
    unsigned short* __restrict__ out,             // [num_tokens, conv_dim], tightly packed
    int num_tokens,
    int conv_dim,
    int kernel_size
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const int state_len = kernel_size - 1;

    float w[CAUSAL_CONV1D_PREFILL_MAX_KERNEL_SIZE];
    float window[CAUSAL_CONV1D_PREFILL_MAX_KERNEL_SIZE];
    const unsigned short* w_row = weight + (size_t)c * (size_t)kernel_size;
    for (int k = 0; k < kernel_size; ++k) {
        w[k] = bf16_bits_to_f32(w_row[k]);
    }
    unsigned short* state_row = conv_state + (size_t)c * (size_t)state_len;
    for (int k = 0; k < state_len; ++k) {
        window[k] = bf16_bits_to_f32(state_row[k]);
    }

    for (int t = 0; t < num_tokens; ++t) {
        const float h_new = bf16_bits_to_f32(src[(size_t)t * (size_t)src_stride + (size_t)src_offset + (size_t)c]);

        float raw = h_new * w[kernel_size - 1];
        for (int k = 0; k < state_len; ++k) {
            raw += window[k] * w[k];
        }

        for (int k = 0; k < state_len - 1; ++k) {
            window[k] = window[k + 1];
        }
        if (state_len > 0) {
            window[state_len - 1] = h_new;
        }

        const float activated = raw / (1.0f + expf(-raw));
        out[(size_t)t * (size_t)conv_dim + (size_t)c] = f32_to_bf16_bits(activated);
    }

    for (int k = 0; k < state_len; ++k) {
        state_row[k] = f32_to_bf16_bits(window[k]);
    }
}

// Plain C entry point. `threads` blocks cover `conv_dim` channels, ONE
// launch for the whole `num_tokens`-token chunk (vs. `num_tokens` real
// launches of `causal_conv1d_update_bf16` before this kernel existed).
extern "C" void launch_causal_conv1d_prefill_bf16(
    const void* src,
    int src_stride,
    int src_offset,
    void* conv_state,
    const void* weight,
    void* out,
    int num_tokens,
    int conv_dim,
    int kernel_size,
    int threads,
    void* stream
) {
    int channel_blocks = (conv_dim + threads - 1) / threads;
    hipLaunchKernelGGL(
        causal_conv1d_prefill_bf16_kernel,
        dim3(channel_blocks), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)src, src_stride, src_offset,
        (unsigned short*)conv_state, (const unsigned short*)weight, (unsigned short*)out,
        num_tokens, conv_dim, kernel_size
    );
}
