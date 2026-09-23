// Real HIP kernel: the single-token cached-decode path of Qwen3.5's GDN
// depthwise causal conv1d (`causal_conv1d_update` in
// transformers/models/qwen3_5/modeling_qwen3_5.py), used by 24 of this
// model's 32 layers on every decode step. Deliberately chosen as this
// project's next feature because it is the EXACT op behind this session's
// earlier research finding (docs/DECISIONS.md §76): on real trained
// weights (not synthetic data), PyTorch's own `F.conv1d(groups=conv_dim)`
// reference kernel gives batch-size-dependent results on this AMD/ROCm
// hardware -- a real ~5.6%-per-verification-step argmax-flip hazard for
// batched speculative verification, root-caused but never fixed (no
// alternative kernel was available in Python). This kernel is a genuine,
// motivated second attempt at the same op, not just another port -- see
// kernels.rs's test suite for whether it reproduces or avoids that
// batch-size dependence.
//
// Formula (matching the real function exactly, kernel_size=4, bias=False,
// activation="silu", real conv_dim=8192 for this model):
//   catted   = concat(conv_state[c, 0..state_len), hidden_states[c])   // length kernel_size
//   raw      = sum_{k=0}^{kernel_size-1} catted[k] * weight[c, k]
//   out[c]   = silu(raw)
//   new_state[c] = catted[1..kernel_size)   // i.e. state shifted left by one, hidden appended
//
// One thread per (batch, channel) pair -- each channel is fully
// independent (depthwise, groups=conv_dim), and the only redundant tap
// count is kernel_size=4, small enough that no shared-memory reduction is
// worth it (unlike rmsnorm.hip's full-row reduction).
//
// UPDATED (§92 performance pass): the original version launched only
// `dim3(batch)` blocks (a single block for this project's real batch=1
// decode case!) with a stride loop covering all conv_dim=8192 channels
// within that ONE block -- correct, but using at most 1 of this GPU's 96
// compute units for the whole kernel. A real profiling pass (§92,
// docs/DECISIONS.md) found GDN layers dominating per-token time (71%)
// and this kernel, while touching relatively few bytes, was one of
// several hand-written kernels launched with far too little block-level
// parallelism to let the GPU hide memory latency across compute units.
// Now launches `ceil(conv_dim/threads)` blocks per batch row (32 blocks
// for this model's real conv_dim=8192 at threads=256), one thread per
// channel directly -- no stride loop needed, since there are now enough
// blocks to cover every channel in one wave.

#include "bf16_utils.hip"

extern "C" __global__ void causal_conv1d_update_bf16_kernel(
    const unsigned short* __restrict__ hidden_states, // [batch, conv_dim]
    unsigned short* __restrict__ conv_state,           // [batch, conv_dim, state_len], updated in place
    const unsigned short* __restrict__ weight,          // [conv_dim, kernel_size]
    unsigned short* __restrict__ out,                    // [batch, conv_dim]
    int conv_dim,
    int kernel_size,
    long long src_stride,   // elements between sequences in hidden_states
    int src_offset          // start of the qkv sub-range within a source row
) {
    const int batch = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const int state_len = kernel_size - 1;

    unsigned short* state_row = conv_state + ((size_t)batch * conv_dim + c) * (size_t)state_len;
    const unsigned short* w_row = weight + (size_t)c * (size_t)kernel_size;
    const unsigned short h_new_bits = hidden_states[(size_t)batch * (size_t)src_stride + (size_t)src_offset + c];
    const float h_new = bf16_bits_to_f32(h_new_bits);

    // catted = [state[0..state_len), h_new]; raw = sum(catted[k] * w[k])
    float raw = h_new * bf16_bits_to_f32(w_row[kernel_size - 1]);
    for (int k = 0; k < state_len; ++k) {
        raw += bf16_bits_to_f32(state_row[k]) * bf16_bits_to_f32(w_row[k]);
    }

    // Shift state left by one, append h_new -- must read state_row[k+1]
    // before overwriting state_row[k], which left-to-right order
    // guarantees (each slot is read once, before it is itself
    // overwritten by the following iteration).
    for (int k = 0; k < state_len - 1; ++k) {
        state_row[k] = state_row[k + 1];
    }
    if (state_len > 0) {
        state_row[state_len - 1] = h_new_bits;
    }

    const float activated = raw / (1.0f + expf(-raw)); // silu
    out[(size_t)batch * conv_dim + c] = f32_to_bf16_bits(activated);
}

// Plain C entry point. `batch` rows, `conv_dim` channels, `kernel_size`
// taps (state_len = kernel_size - 1 per channel). `conv_state` is read AND
// written in place -- callers must not reuse the pre-call buffer contents
// after this returns (matches the real Python function's own
// `conv_state.copy_(...)` in-place-update contract).
// `stream` (§93): see `rmsnorm.hip`'s launcher comment.
extern "C" void launch_causal_conv1d_update_bf16(
    const void* hidden_states,
    void* conv_state,
    const void* weight,
    void* out,
    int batch,
    int conv_dim,
    int kernel_size,
    long long src_stride,
    int src_offset,
    void* stream
) {
    int threads = conv_dim < 256 ? conv_dim : 256;
    int channel_blocks = (conv_dim + threads - 1) / threads;
    hipLaunchKernelGGL(
        causal_conv1d_update_bf16_kernel,
        dim3(channel_blocks, batch), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)hidden_states,
        (unsigned short*)conv_state,
        (const unsigned short*)weight,
        (unsigned short*)out,
        conv_dim, kernel_size, src_stride, src_offset
    );
}
