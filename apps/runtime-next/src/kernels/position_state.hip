// Real HIP kernel: single on-device `int`, incremented by exactly 1.
//
// §93: the last node of a graph-captured decode step. `rope.hip`,
// `kv_cache_append.hip`, and `attention.hip` all read the CURRENT position
// through a device pointer (not a host-passed value) specifically so a HIP
// Graph replay picks up the new value automatically -- this kernel is what
// actually advances that value between replays, entirely on-device, so a
// graph-captured decode loop needs zero host-side writes to keep position
// tracking correct across tokens (only the newly-sampled token id, which is
// inherently host-computed via argmax over logits, needs writing between
// replays).

#include <hip/hip_runtime.h>

extern "C" __global__ void increment_position_kernel(int* position_ptr) {
    *position_ptr += 1;
}

extern "C" void launch_increment_position(int* position_ptr, void* stream) {
    hipLaunchKernelGGL(increment_position_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream, position_ptr);
}
