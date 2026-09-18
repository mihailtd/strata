// Real HIP kernel: §100 (chunked GDN parallel prefill). Solves
// `(I + strictly_lower(ut_system)) @ X = RHS` for `X`, via forward
// substitution -- matching transformers' real
// `torch.linalg.solve_triangular(ut_system, rhs, upper=False,
// unitriangular=True)` call EXACTLY (modeling_qwen3_5.py:395-396):
// `unitriangular=True` means the solver treats the diagonal of
// `ut_system` as an IMPLICIT 1 regardless of what's actually stored there
// (real, nonzero values from `(k_beta @ key^T) * pairwise_decay`'s own
// diagonal -- NOT an error to ignore them, this is the documented
// contract of `unitriangular=True`), and only the STRICTLY lower
// triangle (`j < i`) is read.
//
// Forward substitution: `X[i,:] = RHS[i,:] - sum_{j=0}^{i-1} ut_system[i,j] * X[j,:]`
// -- genuinely sequential over `i` (row `i` needs every earlier row's
// already-solved `X[j,:]`), but fully parallel across the RHS's own
// column dimension (`head_dim` -- one thread per column) AND across
// (head, chunk) pairs (no cross-(head,chunk) dependency at all, unlike
// the LATER real inter-chunk recurrent scan). `X`'s growing rows are kept
// in shared memory (real fp32, for precision during the O(chunk_size^2)
// accumulation -- only rounded to bf16 once, on the final global write,
// matching this crate's "fp32 accumulate, round at the boundary"
// convention everywhere else).

#include "bf16_utils.hip"

extern "C" __global__ void gdn_chunk_utsolve_bf16_kernel(
    const unsigned short* __restrict__ ut_system, // [num_heads, num_chunks, chunk_size, chunk_size]
    const unsigned short* __restrict__ rhs,        // [num_heads, num_chunks, chunk_size, width]
    unsigned short* __restrict__ x_out,             // same shape as rhs
    int num_heads,
    int num_chunks,
    int chunk_size,
    int width
) {
    extern __shared__ float x_shared[]; // [chunk_size, width]

    const int head = blockIdx.x;
    const int chunk = blockIdx.y;
    if (head >= num_heads || chunk >= num_chunks) return;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const unsigned short* ut_base = ut_system + ((size_t)head * num_chunks + chunk) * (size_t)chunk_size * (size_t)chunk_size;
    const unsigned short* rhs_base = rhs + ((size_t)head * num_chunks + chunk) * (size_t)chunk_size * (size_t)width;
    unsigned short* x_base = x_out + ((size_t)head * num_chunks + chunk) * (size_t)chunk_size * (size_t)width;

    for (int i = 0; i < chunk_size; ++i) {
        const unsigned short* ut_row = ut_base + (size_t)i * (size_t)chunk_size; // real row i; only j<i is ever read
        for (int d = tid; d < width; d += stride) {
            float acc = bf16_bits_to_f32(rhs_base[(size_t)i * width + d]);
            for (int j = 0; j < i; ++j) {
                acc -= bf16_bits_to_f32(ut_row[j]) * x_shared[(size_t)j * width + d];
            }
            x_shared[(size_t)i * width + d] = acc;
        }
        // Every thread must see row i's fully-written values before any
        // thread proceeds to row i+1 (which reads x_shared[i,:]).
        __syncthreads();
    }

    for (int idx = tid; idx < chunk_size * width; idx += stride) {
        x_base[idx] = f32_to_bf16_bits(x_shared[idx]);
    }
}

// Plain C entry point. `threads` should be `>= width` for full column
// parallelism (real usage: `width = head_dim`); shared memory sized for
// `chunk_size * width` floats.
extern "C" void launch_gdn_chunk_utsolve_bf16(
    const void* ut_system,
    const void* rhs,
    void* x_out,
    int num_heads,
    int num_chunks,
    int chunk_size,
    int width,
    int threads,
    void* stream
) {
    size_t shmem_bytes = (size_t)chunk_size * (size_t)width * sizeof(float);
    hipLaunchKernelGGL(
        gdn_chunk_utsolve_bf16_kernel,
        dim3(num_heads, num_chunks), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (const unsigned short*)ut_system, (const unsigned short*)rhs, (unsigned short*)x_out,
        num_heads, num_chunks, chunk_size, width
    );
}
