//! Safe HIP runtime foundation: the first real slice of the port.
//!
//! DESIGN, and why it's built this way
//! ------------------------------------
//! This module is the ONLY place in the crate allowed to touch raw HIP FFI
//! or write `unsafe`. Everything above `mod ffi` is safe Rust. This is the
//! "small, curated, auditable unsafe surface" pattern researched at length
//! in `ECOSYSTEM_NOTES.md` before any of this was written (the fearless_simd
//! type-state-gating entry, its "GPU Offload in Rust" `Preload`/`PreloadMut`
//! elaboration, and the NVIDIA CUDA Rust `DisjointSlice`/launch-contract
//! entries) -- not a style choice made up on the spot.
//!
//! No `bindgen`: HIP's real API surface is thousands of functions: pulling
//! all of it in unaudited would be exactly the "large, unaudited unsafe
//! surface" this design is trying to avoid. Only the handful of functions
//! actually used are declared here, by hand, checked against the real
//! headers on this machine (`/opt/rocm/include/hip/hip_runtime_api.h`,
//! `driver_types.h`) at the time of writing.
//!
//! `DeviceBuffer<T>` is the `Drop`-based deterministic GPU memory pattern
//! logged from flodl's release notes: the allocation is freed the instant
//! the buffer goes out of scope, not on a GC's schedule and not via a
//! manual `gc.collect()`/`empty_cache()` dance -- the Python runtime's own
//! `apps/factory/train_all_27b_experts.py` needing exactly that dance
//! between sequential domain-training runs is the concrete motivating
//! example for why this matters (see `apps/factory-next/README.md`).
//!
//! ZERO-MOCK: every function here does the real HIP call. There is no
//! placeholder, no simulated device, no fabricated result anywhere in this
//! file. `hip::tests` below runs against whatever GPU is actually present.

use std::ffi::{CStr, c_int, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::ptr;

/// The only unsafe surface in this crate. Hand-curated against
/// `/opt/rocm/include/hip/hip_runtime_api.h` (ROCm 7.2, this machine).
mod ffi {
    use super::{c_int, c_void};

    unsafe extern "C" {
        pub fn hipGetDeviceCount(count: *mut c_int) -> c_int;
        pub fn hipSetDevice(device_id: c_int) -> c_int;
        pub fn hipDeviceSynchronize() -> c_int;
        pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
        pub fn hipFree(ptr: *mut c_void) -> c_int;
        pub fn hipMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            size_bytes: usize,
            kind: c_int,
        ) -> c_int;
        pub fn hipGetErrorString(error: c_int) -> *const std::ffi::c_char;
        pub fn hipGetLastError() -> c_int;

        // §93: HIP Graph capture/replay -- the "monolithic HIP Graph
        // pipeline" this crate is literally named for (see `main.rs`'s own
        // target string). Feasibility checked here first, isolated, before
        // touching the real decode-loop hot path -- same discipline as the
        // original hipcc-compilation smoke test that de-risked this
        // project's very first .hip kernel (§80).
        pub fn hipStreamBeginCapture(stream: *mut c_void, mode: c_int) -> c_int;
        pub fn hipStreamEndCapture(stream: *mut c_void, graph_out: *mut *mut c_void) -> c_int;
        pub fn hipGraphInstantiate(
            graph_exec_out: *mut *mut c_void,
            graph: *mut c_void,
            error_node: *mut c_void,
            log_buffer: *mut c_void,
            buffer_size: usize,
        ) -> c_int;
        pub fn hipGraphLaunch(graph_exec: *mut c_void, stream: *mut c_void) -> c_int;
        pub fn hipStreamSynchronize(stream: *mut c_void) -> c_int;
        pub fn hipGraphDestroy(graph: *mut c_void) -> c_int;
        pub fn hipGraphExecDestroy(graph_exec: *mut c_void) -> c_int;
        // A real, explicitly-created stream -- required for capture:
        // capturing the null/legacy stream directly was tried first and
        // confirmed (not assumed) to fail with `hipErrorStreamCaptureUnsupported`
        // on this system, matching CUDA/HIP's own documented restriction.
        pub fn hipStreamCreate(stream: *mut *mut c_void) -> c_int;
        pub fn hipStreamDestroy(stream: *mut c_void) -> c_int;
        pub fn hipMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            size_bytes: usize,
            kind: c_int,
            stream: *mut c_void,
        ) -> c_int;
        // §95: real HTTP server -- `DecodeState::reset()` needs to zero
        // GDN's `conv_state`/`recurrent_state` between independent
        // requests (real read-modify-write state, unlike the KV caches,
        // which never need zeroing -- see that method's own doc comment).
        pub fn hipMemset(dst: *mut c_void, value: c_int, size_bytes: usize) -> c_int;
        // §117: real, STREAM-ORDERED, async zero-fill -- `hipMemset`
        // above is synchronous and always targets the legacy default
        // stream, both of which make it real, confirmed-incompatible with
        // HIP Graph capture (capture only records async, stream-ordered
        // operations on the SPECIFIC stream being captured). Needed for
        // bucketed prefill's own real zero-pad step to be capturable.
        pub fn hipMemsetAsync(dst: *mut c_void, value: c_int, size_bytes: usize, stream: *mut c_void) -> c_int;
        // §106: real free/total VRAM query -- used to fail loudly before a
        // model load that won't fit (27B/MoE quantized weights are large
        // enough on this 24GB card that "try it and see" risks an opaque
        // mid-load OOM instead of a clear, actionable error).
        pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
        // §130: pinned (page-locked) host memory -- required for truly async
        // host→device DMA. `hipMemcpyAsync` from pageable host RAM silently
        // falls back to a synchronous copy on ROCm because the DMA engine
        // cannot directly address pageable pages. Allocating the source
        // buffer via `hipHostMalloc` pins it: the driver registers it with
        // the IOMMU, giving the DMA engine a stable physical address it can
        // access without CPU involvement, so `hipMemcpyAsync` truly overlaps
        // with other work on the stream. `hipHostFree` releases both the
        // host VA and the IOMMU registration in one call.
        pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_int) -> c_int;
        pub fn hipHostFree(ptr: *mut c_void) -> c_int;
    }

    // hipMemcpyKind (driver_types.h) -- the two directions this crate uses.
    pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
    pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
    // hipStreamCaptureMode (hip_runtime_api.h)
    pub const HIP_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;
    pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
    // §130: hipHostMallocDefault -- standard pinned host memory, no special
    // mapping flags. The DMA engine can address it directly; the host can
    // read/write it normally. hipHostMalloc flags field (hip_runtime_api.h).
    pub const HIP_HOST_MALLOC_DEFAULT: c_int = 0;
}

/// A HIP runtime error, carrying both the numeric code and the string HIP
/// itself provides for it (via `hipGetErrorString`, not a hand-maintained
/// message table that could drift from what the runtime actually means).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipError {
    pub code: i32,
}

impl HipError {
    fn from_code(code: c_int) -> Result<(), HipError> {
        if code == 0 {
            Ok(())
        } else {
            Err(HipError { code })
        }
    }

    pub fn message(&self) -> String {
        // SAFETY: hipGetErrorString returns a pointer to a static,
        // null-terminated string owned by the HIP runtime for the lifetime
        // of the process; it is never null for any hipError_t value.
        unsafe {
            let ptr = ffi::hipGetErrorString(self.code);
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HIP error {}: {}", self.code, self.message())
    }
}

impl std::error::Error for HipError {}

/// Number of HIP-visible devices on this machine. A real call, not a
/// cached/assumed value -- this is meant to be called once at startup, not
/// in a hot loop.
pub fn device_count() -> Result<i32, HipError> {
    let mut count: c_int = 0;
    // SAFETY: `count` is a valid, aligned, writable `c_int` for the
    // duration of this call; HIP writes exactly one `c_int` through it.
    let code = unsafe { ffi::hipGetDeviceCount(&mut count) };
    HipError::from_code(code)?;
    Ok(count)
}

/// Real free/total VRAM on the currently-selected device, in bytes
/// (`hipMemGetInfo`) -- not estimated from the model's own weight sizes,
/// since other processes (or a prior load in the same process) can hold
/// real allocations this crate doesn't know about otherwise. Meant to be
/// checked before a large load (27B+/MoE), not in a hot loop.
pub fn mem_info() -> Result<(usize, usize), HipError> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: `free`/`total` are valid, aligned, writable `usize`s for the
    // duration of this call; HIP writes exactly one `usize` through each.
    let code = unsafe { ffi::hipMemGetInfo(&mut free, &mut total) };
    HipError::from_code(code)?;
    Ok((free, total))
}

/// Selects the active HIP device for the calling thread's subsequent HIP
/// calls. Real HIP call, not a no-op placeholder.
pub fn set_device(device_id: i32) -> Result<(), HipError> {
    // SAFETY: hipSetDevice takes a plain integer; no pointer/lifetime
    // invariants to uphold.
    let code = unsafe { ffi::hipSetDevice(device_id) };
    HipError::from_code(code)
}

/// Blocks the calling thread until all previously queued HIP work on the
/// active device has completed.
pub fn device_synchronize() -> Result<(), HipError> {
    // SAFETY: no arguments, no pointers.
    let code = unsafe { ffi::hipDeviceSynchronize() };
    HipError::from_code(code)
}

/// Checks HIP's "last error" state -- specifically, the kind of error an
/// invalid kernel launch CONFIGURATION (too many threads per block, an
/// invalid grid shape) surfaces immediately, synchronously, before the
/// kernel would even run. Call this right after every kernel launch, not
/// only after a later `device_synchronize()`.
///
/// Found the hard way why this matters: `embedding.hip`'s first version
/// launched 2560 threads per block (`hidden_size`, no cap) -- over AMD
/// hardware's 1024-thread-per-block limit. The launch did not surface as
/// an error through `device_synchronize()` alone in every case; the
/// kernel's own correctness test failing (against real, independently-
/// checked data) is what actually caught it. This function exists so the
/// NEXT invalid launch configuration is a loud, immediate `HipError`
/// instead of relying on every kernel having an equally strict correctness
/// test to notice silently wrong output -- the same "NO SILENT FALLBACKS"
/// principle `apps/runtime-ipwf/fused_norm.py` already established for
/// this project's Python runtime, applied here for the first time on the
/// Rust side.
pub fn check_last_error() -> Result<(), HipError> {
    // SAFETY: no arguments, no pointers; reads and clears HIP's internal
    // last-error state for the calling thread.
    let code = unsafe { ffi::hipGetLastError() };
    HipError::from_code(code)
}

/// A GPU-resident buffer of `len` elements of `T`, owning its own
/// `hipMalloc`'d allocation. `Drop` calls `hipFree` deterministically --
/// see the module doc comment for why this specific property is the point.
///
/// `T` is bounded by `Copy` because this buffer moves bytes via `hipMemcpy`,
/// which knows nothing about `Drop` glue, invariants, or non-`Copy` types on
/// either side of the copy -- the same reasoning `Preload`/`PreloadMut`
/// (`ECOSYSTEM_NOTES.md`) is built on: the host and device copies are
/// genuinely different objects at different addresses, not a shared
/// reference, so only plain, bit-copyable data belongs on either side of
/// this boundary today. `T` is also bounded by `Send`/`Sync` — false safety
/// otherwise, since the buffer can outlive the thread that allocated it.
pub struct DeviceBuffer<T: Copy> {
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: the underlying HIP allocation is not tied to any host thread; HIP
// itself is thread-safe for the operations this type exposes (alloc/free/
// memcpy against a fixed device pointer).
unsafe impl<T: Copy + Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocates real, uninitialized GPU memory for `len` elements of `T`.
    /// Returns `Err` on `len == 0` rather than making a zero-byte
    /// `hipMalloc` call whose semantics aren't needed here.
    pub fn alloc(len: usize) -> Result<Self, HipError> {
        if len == 0 {
            return Err(HipError { code: 1 }); // hipErrorInvalidValue
        }
        let size_bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("DeviceBuffer::alloc: len * size_of::<T>() overflowed usize");

        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: `ptr` is a valid, aligned, writable `*mut c_void` for the
        // duration of this call; HIP writes exactly one pointer through it,
        // or leaves it untouched and returns a nonzero error code (checked
        // immediately below).
        let code = unsafe { ffi::hipMalloc(&mut ptr, size_bytes) };
        HipError::from_code(code)?;
        debug_assert!(
            !ptr.is_null(),
            "hipMalloc returned success with a null pointer"
        );

        Ok(DeviceBuffer {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer, for passing to a custom kernel launcher (see
    /// `src/kernels.rs`). This is the one intentional escape hatch in this
    /// type: kernel launches need a real device address, and Rust's borrow
    /// checker cannot see across the FFI boundary into what a `.hip` kernel
    /// does with it. The caller (always this crate's own `kernels` module)
    /// is responsible for launching a kernel that respects `self.len()`
    /// elements and the buffer's real lifetime -- the same trust boundary
    /// as any other kernel-launch API (CUDA/HIP itself, `cuda-oxide`'s
    /// `#[launch_contract]`, etc.), not a gap specific to this type.
    pub fn as_device_ptr(&self) -> *const c_void {
        self.ptr
    }

    /// Mutable counterpart of `as_device_ptr` -- for buffers a kernel
    /// writes into.
    pub fn as_device_ptr_mut(&mut self) -> *mut c_void {
        self.ptr
    }

    /// A read-only view into this buffer starting `elem_offset` elements
    /// in -- for splitting one contiguous allocation into several
    /// sub-tensors without a copy (e.g. `in_proj_qkv`'s output is really
    /// three back-to-back tensors: query, key, value). The caller is
    /// responsible for staying within `[elem_offset, self.len())`, same
    /// trust boundary as `as_device_ptr` itself.
    pub fn as_device_ptr_at(&self, elem_offset: usize) -> *const c_void {
        debug_assert!(
            elem_offset <= self.len,
            "as_device_ptr_at: offset {elem_offset} out of bounds for length {}",
            self.len
        );
        unsafe { (self.ptr as *const T).add(elem_offset) as *const c_void }
    }

    /// Mutable counterpart of `as_device_ptr_at` -- §96: a batched prefill
    /// chunk's per-token inner loops (rope/kv_cache_append/attention_decode/
    /// causal_conv1d_update/gdn_gate_beta/gdn_recurrent_decode) each write
    /// one row's worth of a `[T, dim]` scratch buffer per iteration; this is
    /// that write target.
    pub fn as_device_ptr_at_mut(&mut self, elem_offset: usize) -> *mut c_void {
        debug_assert!(
            elem_offset <= self.len,
            "as_device_ptr_at_mut: offset {elem_offset} out of bounds for length {}",
            self.len
        );
        unsafe { (self.ptr as *mut T).add(elem_offset) as *mut c_void }
    }

    fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Copies `host_data` into this buffer. `host_data.len()` must equal
    /// `self.len()` -- checked, not assumed, matching this project's own
    /// Zero-Mock Invariant that a real check beats an implicit contract.
    pub fn copy_from_host(&mut self, host_data: &[T]) -> Result<(), HipError> {
        assert_eq!(
            host_data.len(),
            self.len,
            "DeviceBuffer::copy_from_host: length mismatch ({} host vs {} device)",
            host_data.len(),
            self.len
        );
        // SAFETY: `self.ptr` is a live `hipMalloc`'d allocation of at least
        // `self.byte_len()` bytes (invariant maintained by `alloc` and never
        // mutated elsewhere); `host_data` is a valid, readable slice of
        // exactly that many bytes for `T: Copy`. Direction is host->device.
        let code = unsafe {
            ffi::hipMemcpy(
                self.ptr,
                host_data.as_ptr() as *const c_void,
                self.byte_len(),
                ffi::HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        HipError::from_code(code)
    }

    /// §96: copies `host_data` into the FIRST `host_data.len()` elements of
    /// this buffer -- `host_data.len()` must be `<= self.len()` (not equal,
    /// unlike `copy_from_host`). Needed for `PrefillScratch`'s
    /// `token_ids_dev`/`position_buf`, allocated once at the fixed
    /// `MAX_PREFILL_CHUNK` capacity but written with real, varying-length
    /// chunk data (a real prompt is essentially never exactly
    /// `MAX_PREFILL_CHUNK` tokens long) -- `copy_from_host`'s exact-length
    /// check exists specifically to catch accidental short/long writes
    /// where a full-buffer copy really was intended; this method exists for
    /// the genuinely different case where a partial write is the real,
    /// correct intent.
    pub fn copy_from_host_prefix(&mut self, host_data: &[T]) -> Result<(), HipError> {
        assert!(
            host_data.len() <= self.len,
            "DeviceBuffer::copy_from_host_prefix: {} host elements exceed device capacity {}",
            host_data.len(),
            self.len
        );
        let byte_len = host_data.len() * std::mem::size_of::<T>();
        // SAFETY: `self.ptr` is a live `hipMalloc`'d allocation of at least
        // `self.byte_len()` bytes, and `byte_len <= self.byte_len()` by the
        // assert above; `host_data` is a valid, readable slice of exactly
        // `byte_len` bytes for `T: Copy`. Direction is host->device.
        let code = unsafe {
            ffi::hipMemcpy(
                self.ptr,
                host_data.as_ptr() as *const c_void,
                byte_len,
                ffi::HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        HipError::from_code(code)
    }

    /// Copies another `DeviceBuffer`'s contents into this one, entirely on
    /// the GPU (no host round-trip). `src.len()` must equal `self.len()`.
    /// §100 (LoRA in-place weight folding): the real pristine-weight
    /// restore step (`W_active <- W_0`) -- a device-to-device copy is the
    /// correct primitive here (both buffers already live in VRAM; a
    /// host round-trip would be pure waste, and per `TODO_LORA_SWAP.md`'s
    /// own real bandwidth math, ~200x slower for a multi-GB restore).
    pub fn copy_from_device(&mut self, src: &DeviceBuffer<T>) -> Result<(), HipError> {
        assert_eq!(
            src.len,
            self.len,
            "DeviceBuffer::copy_from_device: length mismatch ({} src vs {} dst)",
            src.len,
            self.len
        );
        // SAFETY: both `self.ptr` and `src.ptr` are live `hipMalloc`'d
        // allocations of at least `self.byte_len()` bytes (equal lengths
        // checked above); `hipMemcpy` with `HIP_MEMCPY_DEVICE_TO_DEVICE`
        // is a well-defined GPU-to-GPU copy for any `T: Copy`.
        let code = unsafe { ffi::hipMemcpy(self.ptr, src.ptr as *const c_void, self.byte_len(), ffi::HIP_MEMCPY_DEVICE_TO_DEVICE) };
        HipError::from_code(code)
    }

    /// §137: like `copy_from_device`, but reads starting at `src_offset`
    /// elements into `src` rather than `src`'s own start -- needed to
    /// pull a single row (e.g. one chunk position's hidden state) out of
    /// a larger, multi-row buffer (e.g. a preserved `[k, HIDDEN_SIZE]`
    /// verify-chunk hidden-state buffer) into a single-row destination.
    /// Ordinary blocking `hipMemcpy`: called from eager code between
    /// rounds, never inside a captured graph region.
    pub fn copy_from_device_offset(&mut self, src: &DeviceBuffer<T>, src_offset: usize) -> Result<(), HipError> {
        assert!(
            src_offset + self.len <= src.len,
            "DeviceBuffer::copy_from_device_offset: offset {} + dst len {} exceeds src capacity {}",
            src_offset, self.len, src.len
        );
        // SAFETY: `self.ptr` is a live allocation of at least
        // `self.byte_len()` bytes; `src.ptr` offset by `src_offset`
        // elements is a valid read start with at least `self.byte_len()`
        // bytes remaining (checked above); `hipMemcpy` with
        // `HIP_MEMCPY_DEVICE_TO_DEVICE` is well-defined for any `T: Copy`.
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset * std::mem::size_of::<T>()) as *const c_void };
        let code = unsafe { ffi::hipMemcpy(self.ptr, src_ptr, self.byte_len(), ffi::HIP_MEMCPY_DEVICE_TO_DEVICE) };
        HipError::from_code(code)
    }

    /// §137: real, STREAM-ORDERED async device-to-device copy -- the
    /// async counterpart of `copy_from_device` (that one uses the
    /// blocking `hipMemcpy`, safe for ordinary eager code but NOT
    /// graph-capture-safe: a synchronous host-blocking call cannot be
    /// recorded into a HIP graph). Needed so a graph-captured verify
    /// step can preserve a real intermediate buffer (e.g. the PRE-final-
    /// norm per-position hidden state a captured verify-chunk graph
    /// would otherwise discard once it's done computing the POST-norm
    /// logits) into a caller-provided output buffer, as part of the
    /// SAME captured, unconditional kernel sequence -- not a follow-up
    /// eager call after the graph replay. The `count`-limited variant
    /// below is the one real call site actually needs (a bucket's real
    /// row count is always `<=` the source's full allocated capacity):
    /// same relationship as `copy_from_host_prefix` is to a full-buffer
    /// host copy. Needed because a captured verify-chunk's own internal
    /// hidden-state buffer (`PrefillScratch::hidden_a`/`hidden_b`) is
    /// allocated once at the fixed `MAX_PREFILL_CHUNK` capacity, but a
    /// given bucket's real captured sequence only ever produces `k` valid
    /// rows (`k <= MAX_PREFILL_CHUNK`) -- the caller's own output buffer
    /// is correctly sized to that real `k`, not the source's full
    /// capacity, so an exact-length copy would over-read past what the
    /// caller allocated (or under-specify what the source actually
    /// holds). Copies `count` elements from the start of `src` into the
    /// start of `self`; `count` must not exceed either buffer's capacity.
    pub fn copy_from_device_async_prefix(&mut self, src: &DeviceBuffer<T>, count: usize, stream: *mut c_void) -> Result<(), HipError> {
        assert!(
            count <= src.len && count <= self.len,
            "DeviceBuffer::copy_from_device_async_prefix: count {} exceeds src capacity {} or dst capacity {}",
            count, src.len, self.len
        );
        let byte_len = count * std::mem::size_of::<T>();
        // SAFETY: both `self.ptr`/`src.ptr` are live `hipMalloc`'d
        // allocations of at least `count * size_of::<T>()` bytes (checked
        // above); `hipMemcpyAsync` with `HIP_MEMCPY_DEVICE_TO_DEVICE` is a
        // well-defined, stream-ordered GPU-to-GPU copy for any `T: Copy`.
        let code = unsafe { ffi::hipMemcpyAsync(self.ptr, src.ptr as *const c_void, byte_len, ffi::HIP_MEMCPY_DEVICE_TO_DEVICE, stream) };
        HipError::from_code(code)
    }

    /// Copies this buffer's contents into `host_data`. `host_data.len()`
    /// must equal `self.len()`.
    pub fn copy_to_host(&self, host_data: &mut [T]) -> Result<(), HipError> {
        assert_eq!(
            host_data.len(),
            self.len,
            "DeviceBuffer::copy_to_host: length mismatch ({} host vs {} device)",
            host_data.len(),
            self.len
        );
        // SAFETY: symmetric to `copy_from_host`; direction is device->host.
        let code = unsafe {
            ffi::hipMemcpy(
                host_data.as_mut_ptr() as *mut c_void,
                self.ptr,
                self.byte_len(),
                ffi::HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        HipError::from_code(code)
    }

    /// Zeroes this buffer's entire contents in place, synchronously.
    /// §95: `DecodeState::reset()` uses this to clear GDN's real
    /// read-modify-write recurrent state between independent HTTP
    /// requests -- unlike a host-side zero `Vec` + `copy_from_host`, this
    /// never touches host memory at all.
    pub fn fill_zero(&mut self) -> Result<(), HipError> {
        // SAFETY: `self.ptr` is a live `hipMalloc`'d allocation of at least
        // `self.byte_len()` bytes; `hipMemset` writes exactly that many
        // bytes to the single byte value 0, a well-defined operation for
        // any `T: Copy` (all-zero-bytes is the additive identity for both
        // `f32` and `u16`-as-bf16-bits, the only element types this crate
        // ever zeroes).
        let code = unsafe { ffi::hipMemset(self.ptr, 0, self.byte_len()) };
        HipError::from_code(code)
    }

    /// Zeroes only the TAIL of this buffer, `[elem_offset, self.len())`,
    /// synchronously. §100 (chunked GDN prefill): real chunk padding --
    /// the last chunk of a `num_tokens` not evenly divisible by
    /// `GDN_CHUNK_SIZE` needs its `[num_tokens, padded_len)` tail zeroed
    /// (matching the real reference's own `F.pad(..., 0)`) so the fake
    /// padding positions become pure no-op identity steps (zero
    /// key/value/beta, zero log-decay) instead of leaking stale data from
    /// a previous call into this call's real recurrent-state update.
    #[allow(dead_code)]
    pub fn fill_zero_from(&mut self, elem_offset: usize) -> Result<(), HipError> {
        debug_assert!(
            elem_offset <= self.len,
            "fill_zero_from: offset {elem_offset} out of bounds for length {}",
            self.len
        );
        let byte_offset = elem_offset * std::mem::size_of::<T>();
        let remaining_bytes = self.byte_len() - byte_offset;
        // SAFETY: same reasoning as `fill_zero`, restricted to the real
        // in-bounds sub-range `[elem_offset, self.len())` (checked above).
        let code = unsafe { ffi::hipMemset((self.ptr as *mut u8).add(byte_offset) as *mut c_void, 0, remaining_bytes) };
        HipError::from_code(code)
    }

    /// §117: real, STREAM-ORDERED async counterpart of `fill_zero_from`
    /// above -- same real zeroing semantics (`[elem_offset, self.len())`),
    /// queued on the given real `stream` via `hipMemsetAsync` instead of
    /// blocking the host on the legacy default stream. The one real
    /// difference that matters: this IS capturable into a HIP Graph
    /// (`fill_zero_from`'s own `hipMemset` is not -- see that FFI decl's
    /// own doc comment).
    pub fn fill_zero_from_async(&mut self, elem_offset: usize, stream: *mut c_void) -> Result<(), HipError> {
        debug_assert!(
            elem_offset <= self.len,
            "fill_zero_from_async: offset {elem_offset} out of bounds for length {}",
            self.len
        );
        let byte_offset = elem_offset * std::mem::size_of::<T>();
        let remaining_bytes = self.byte_len() - byte_offset;
        // SAFETY: same reasoning as `fill_zero_from` above.
        let code = unsafe { ffi::hipMemsetAsync((self.ptr as *mut u8).add(byte_offset) as *mut c_void, 0, remaining_bytes, stream) };
        HipError::from_code(code)
    }

    /// §130: real, STREAM-ORDERED async counterpart of `fill_zero` --
    /// zeroes this entire buffer, queued on `stream` via `hipMemsetAsync`.
    /// Same HIP-Graph-capturable invariant as `fill_zero_from_async`; unlike
    /// `fill_zero`'s synchronous `hipMemset`, this returns immediately and
    /// the GPU work proceeds in parallel with subsequent host operations
    /// (or other async ops on the same stream).
    pub fn fill_zero_async(&mut self, stream: *mut c_void) -> Result<(), HipError> {
        // SAFETY: `self.ptr` is a live `hipMalloc`'d allocation of at least
        // `self.byte_len()` bytes; `hipMemsetAsync` writes exactly that many
        // zero bytes stream-ordered on `stream`, safe for any `T: Copy`.
        let code = unsafe { ffi::hipMemsetAsync(self.ptr, 0, self.byte_len(), stream) };
        HipError::from_code(code)
    }

    /// §130: real, STREAM-ORDERED async host→device copy -- the async
    /// counterpart of `copy_from_host`. `host_data` MUST point into a
    /// `PinnedBuffer` (page-locked host memory allocated via `hipHostMalloc`)
    /// for this to be truly asynchronous on ROCm. Passing pageable host RAM
    /// is not rejected by the runtime but silently serialises the copy (the
    /// driver pins the pages internally, one synchronous bounce per call).
    /// Callers are responsible for keeping `host_data` alive until the
    /// stream's work has completed (i.e., until after `stream.synchronize()`).
    pub fn copy_from_host_async(&mut self, host_data: &[T], stream: *mut c_void) -> Result<(), HipError> {
        assert!(
            host_data.len() <= self.len,
            "DeviceBuffer::copy_from_host_async: host data exceeds device buffer capacity ({} host vs {} device)",
            host_data.len(),
            self.len
        );
        if host_data.is_empty() {
            return Ok(());
        }
        let byte_len = host_data
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .expect("DeviceBuffer::copy_from_host_async: byte_len overflow");
        // SAFETY: `self.ptr` is a live `hipMalloc`'d device allocation of at least `byte_len` bytes;
        // `host_data` is a valid, readable slice for its lifetime (caller
        // upholds liveness until stream sync -- documented in the fn doc);
        // direction is host→device.
        let code = unsafe {
            ffi::hipMemcpyAsync(
                self.ptr,
                host_data.as_ptr() as *const c_void,
                byte_len,
                ffi::HIP_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        };
        HipError::from_code(code)
    }

    /// §135: real, STREAM-ORDERED async host→device copy, writing
    /// starting at `elem_offset` instead of the start of the buffer --
    /// the offset counterpart of `copy_from_host_async`, needed for
    /// writing one real slot's data into its own sub-range of a larger,
    /// shared fused buffer (`QuantLoraMidFused`) without disturbing any
    /// other slot's data living in the rest of that same buffer.
    pub fn copy_from_host_async_at(&mut self, host_data: &[T], elem_offset: usize, stream: *mut c_void) -> Result<(), HipError> {
        assert!(
            elem_offset + host_data.len() <= self.len,
            "DeviceBuffer::copy_from_host_async_at: offset {elem_offset} + host data {} exceeds device buffer capacity {}",
            host_data.len(),
            self.len
        );
        if host_data.is_empty() {
            return Ok(());
        }
        let byte_len = host_data
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .expect("DeviceBuffer::copy_from_host_async_at: byte_len overflow");
        let byte_offset = elem_offset * std::mem::size_of::<T>();
        // SAFETY: bounds-checked above (`elem_offset + host_data.len() <=
        // self.len`); `self.ptr` is a live `hipMalloc`'d allocation of at
        // least `self.byte_len()` bytes, so `self.ptr + byte_offset` for
        // `byte_len` bytes stays within that allocation; `host_data` is a
        // valid, readable slice for its lifetime (caller upholds liveness
        // until stream sync, same contract as `copy_from_host_async`).
        let code = unsafe {
            ffi::hipMemcpyAsync(
                (self.ptr as *mut u8).add(byte_offset) as *mut c_void,
                host_data.as_ptr() as *const c_void,
                byte_len,
                ffi::HIP_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        };
        HipError::from_code(code)
    }

    /// §135: real, STREAM-ORDERED async zero-fill of a BOUNDED range
    /// `[start_elem, end_elem)` -- unlike `fill_zero_from_async` (which
    /// zeroes from an offset to the END of the buffer), this stops at
    /// `end_elem`, so it's safe to use on a shared, multi-slot fused
    /// buffer where zeroing one slot's tail must never touch a
    /// DIFFERENT slot's data living later in the same buffer.
    pub fn fill_zero_range_async(&mut self, start_elem: usize, end_elem: usize, stream: *mut c_void) -> Result<(), HipError> {
        debug_assert!(
            start_elem <= end_elem && end_elem <= self.len,
            "fill_zero_range_async: invalid range [{start_elem}, {end_elem}) for length {}",
            self.len
        );
        if start_elem == end_elem {
            return Ok(());
        }
        let byte_offset = start_elem * std::mem::size_of::<T>();
        let byte_len = (end_elem - start_elem) * std::mem::size_of::<T>();
        // SAFETY: bounds-checked above; `self.ptr + byte_offset` for
        // `byte_len` bytes stays within the live `hipMalloc`'d allocation.
        let code = unsafe { ffi::hipMemsetAsync((self.ptr as *mut u8).add(byte_offset) as *mut c_void, 0, byte_len, stream) };
        HipError::from_code(code)
    }

}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was returned by a successful `hipMalloc` in
        // `alloc` and has not been freed elsewhere -- `DeviceBuffer` is the
        // sole owner (no `Clone` impl) and this is the only `Drop`.
        // Deliberately ignores the return code: `Drop` cannot propagate an
        // error, and a failing `hipFree` here means the device context is
        // already in a bad state that no caller of `drop` could act on.
        unsafe {
            ffi::hipFree(self.ptr);
        }
    }
}

/// A real, explicitly-created HIP stream -- required for graph capture
/// (§93: capturing the null/legacy stream directly is rejected with
/// `hipErrorStreamCaptureUnsupported`, confirmed on this system). RAII,
/// same pattern as `DeviceBuffer`: `Drop` calls `hipStreamDestroy`
/// deterministically.
pub struct Stream {
    raw: *mut c_void,
}

// SAFETY: a HIP stream handle is not tied to any host thread; HIP's stream
// API is safe to call from any thread against a fixed stream handle.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// Creates a real, non-default HIP stream via `hipStreamCreate`.
    pub fn create() -> Result<Self, HipError> {
        let mut raw: *mut c_void = ptr::null_mut();
        // SAFETY: `raw` is a valid, aligned, writable pointer for the
        // duration of this call; HIP writes exactly one stream handle
        // through it, or leaves it untouched and returns a nonzero error
        // code (checked immediately below).
        let code = unsafe { ffi::hipStreamCreate(&mut raw) };
        HipError::from_code(code)?;
        Ok(Stream { raw })
    }

    /// Raw stream handle, for passing to a kernel launcher or hipBLAS call
    /// that takes an explicit stream. Same trust boundary as
    /// `DeviceBuffer::as_device_ptr`: the caller is responsible for the
    /// handle's real lifetime, which this type upholds via `Drop`.
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    /// Blocks the calling thread until all work queued on this stream
    /// (including a replayed graph's nodes) has completed.
    pub fn synchronize(&self) -> Result<(), HipError> {
        // SAFETY: `self.raw` is a live stream handle for the whole call.
        let code = unsafe { ffi::hipStreamSynchronize(self.raw) };
        HipError::from_code(code)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: `self.raw` was returned by a successful `hipStreamCreate`
        // and has not been destroyed elsewhere -- `Stream` is the sole
        // owner (no `Clone` impl) and this is the only `Drop`. Return code
        // ignored for the same reason as `DeviceBuffer::drop`.
        unsafe {
            ffi::hipStreamDestroy(self.raw);
        }
    }
}

/// §130: page-locked (pinned) host memory, allocated via `hipHostMalloc`.
///
/// This is the mandatory source type for a truly-asynchronous
/// host→device `hipMemcpyAsync` on ROCm/AMD. When the source buffer is
/// ordinary pageable `Vec` memory, the ROCm driver must pin the pages
/// internally before the DMA engine can access them -- an implicit,
/// synchronous bounce that serialises every `hipMemcpyAsync` call back to
/// the behaviour of plain `hipMemcpy`. `hipHostMalloc` registers the
/// allocation with the IOMMU at allocation time (not at copy time), giving
/// the DMA engine a stable physical address it can read without any CPU
/// involvement: `hipMemcpyAsync` genuinely returns before the transfer
/// completes, and multiple transfers to different device buffers can overlap
/// on the same stream.
///
/// `T` is bounded by `Copy` for the same reason `DeviceBuffer<T>` is.
/// RAII: `Drop` calls `hipHostFree` deterministically, releasing both
/// the host VA and the IOMMU registration.
pub struct PinnedBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: a pinned allocation is not thread-affine; concurrent access from
// multiple threads is the caller's problem (same contract as DeviceBuffer).
unsafe impl<T: Copy + Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for PinnedBuffer<T> {}

impl<T: Copy> PinnedBuffer<T> {
    /// Allocates `len` elements of pinned host memory (uninitialized).
    /// Fails with a `HipError` if `hipHostMalloc` returns a non-zero code
    /// or a null pointer on success (which should never happen in practice
    /// but is checked so callers can propagate a real error rather than
    /// silently dereferencing null).
    pub fn alloc(len: usize) -> Result<Self, HipError> {
        let size_bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("PinnedBuffer::alloc: len * size_of::<T>() overflowed usize");
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: `ptr` is a valid, aligned writable pointer for the
        // duration of this call; HIP writes exactly one host pointer through
        // it, or leaves it unchanged and returns a nonzero error code.
        let code = unsafe { ffi::hipHostMalloc(&mut ptr, size_bytes, ffi::HIP_HOST_MALLOC_DEFAULT) };
        HipError::from_code(code)?;
        assert!(!ptr.is_null(), "hipHostMalloc returned success with a null pointer");
        Ok(PinnedBuffer { ptr: ptr as *mut T, len, _marker: PhantomData })
    }

    /// Returns the number of `T`-elements this buffer holds.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns a slice view of the pinned host memory. The bytes are
    /// uninitialised until the caller writes to them.
    ///
    /// # Safety
    /// The caller must not read elements that have not been written since
    /// allocation (or since the last `fill` / write). Reading uninitialised
    /// memory is UB in Rust regardless of the backing allocator.
    pub unsafe fn as_slice_uninit(&self) -> &[T] {
        // SAFETY: `self.ptr` is a live, aligned, readable pinned allocation
        // of exactly `self.len` elements; the caller has asserted via the
        // `unsafe` annotation that the elements have been initialised.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a mutable slice over the entire pinned buffer. The caller
    /// can use this to write values before issuing an async copy.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `self.ptr` is a live, aligned, writable pinned allocation
        // of exactly `self.len` elements; `PinnedBuffer` is the sole owner.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Copies `src` into this buffer (host-side, synchronous -- just a
    /// memcpy into pinned RAM). Call this before issuing an async device
    /// copy via `DeviceBuffer::copy_from_host_async`.
    #[allow(dead_code)]
    pub fn copy_from_slice(&mut self, src: &[T]) {
        assert_eq!(
            src.len(), self.len,
            "PinnedBuffer::copy_from_slice: length mismatch ({} src vs {} dst)",
            src.len(), self.len
        );
        // SAFETY: `self.ptr` is a live, aligned pinned allocation of exactly
        // `self.len` elements; `src` is a valid readable slice of the same
        // length; `T: Copy` so there are no Drop concerns on the overwritten
        // destination slots.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, self.len);
        }
    }
}

impl<T: Copy> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was returned by a successful `hipHostMalloc`
        // and has not been freed elsewhere -- `PinnedBuffer` is the sole
        // owner (no `Clone` impl) and this is the only `Drop`. Return code
        // deliberately ignored (same rationale as `DeviceBuffer::drop`).
        unsafe {
            ffi::hipHostFree(self.ptr as *mut c_void);
        }
    }
}

/// Begins recording every HIP operation subsequently issued on `stream`
/// into a graph template, instead of executing them immediately. Pair with
/// `end_capture` to finish recording and get a replayable `GraphExec`.
pub fn begin_capture(stream: &Stream) -> Result<(), HipError> {
    // SAFETY: `stream.raw` is a live, real stream handle for the duration
    // of this call.
    let code = unsafe { ffi::hipStreamBeginCapture(stream.raw, ffi::HIP_STREAM_CAPTURE_MODE_GLOBAL) };
    HipError::from_code(code)
}

/// Ends recording on `stream` (started by `begin_capture`), then
/// instantiates the recorded template into a real, replayable `GraphExec`.
/// The intermediate `hipGraph_t` template is destroyed here -- callers only
/// need the instantiated executable graph, never the template itself.
pub fn end_capture(stream: &Stream) -> Result<GraphExec, HipError> {
    let mut graph: *mut c_void = ptr::null_mut();
    // SAFETY: `stream.raw` is a live stream handle that `begin_capture`
    // already put into capture mode; `graph` is a valid, writable pointer
    // for HIP to write the resulting `hipGraph_t` handle through.
    let code = unsafe { ffi::hipStreamEndCapture(stream.raw, &mut graph) };
    HipError::from_code(code)?;

    let mut graph_exec: *mut c_void = ptr::null_mut();
    // SAFETY: `graph` is the live template handle just produced above;
    // `graph_exec` is a valid, writable pointer. `error_node`/`log_buffer`
    // null -- this crate doesn't need instantiation diagnostics, just
    // success/failure via the returned status code.
    let code = unsafe {
        ffi::hipGraphInstantiate(&mut graph_exec, graph, ptr::null_mut(), ptr::null_mut(), 0)
    };
    let instantiate_result = HipError::from_code(code);

    // SAFETY: `graph` is a live template handle; freeing it here (whether
    // instantiation succeeded or not) does not affect `graph_exec`, which
    // is an independent, already-baked copy.
    unsafe {
        ffi::hipGraphDestroy(graph);
    }
    instantiate_result?;

    Ok(GraphExec { raw: graph_exec })
}

/// A real, instantiated, replayable HIP graph -- the product of
/// `begin_capture`/`end_capture`. RAII: `Drop` calls `hipGraphExecDestroy`.
pub struct GraphExec {
    raw: *mut c_void,
}

unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

impl GraphExec {
    /// Replays every captured operation, in the same order, against
    /// whatever the captured buffers' CURRENT contents are (not what they
    /// held at capture time) -- the whole point of graph replay. Does not
    /// synchronize; call `stream.synchronize()` (or queue more work) after,
    /// same as any other stream-ordered launch in this crate.
    pub fn launch(&self, stream: &Stream) -> Result<(), HipError> {
        // SAFETY: `self.raw` is a live, instantiated graph-exec handle;
        // `stream.raw` is a live stream handle.
        let code = unsafe { ffi::hipGraphLaunch(self.raw, stream.raw) };
        HipError::from_code(code)
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        // SAFETY: `self.raw` was returned by a successful `hipGraphInstantiate`
        // in `end_capture` and has not been destroyed elsewhere -- sole
        // owner (no `Clone` impl), only `Drop`. Return code ignored, same
        // reasoning as `DeviceBuffer::drop`.
        unsafe {
            ffi::hipGraphExecDestroy(self.raw);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real smoke test: queries the actual device count on this machine.
    /// No live GPU is required to PASS this test, but it does make a real
    /// HIP call either way -- a `hipErrorNoDevice`-class result is a valid,
    /// checked-for outcome, not a crash or a silent skip.
    #[test]
    fn real_device_count_is_queryable() {
        match device_count() {
            Ok(n) => assert!(n >= 0, "device count must be non-negative, got {n}"),
            Err(e) => {
                // Only acceptable failure on a machine with no visible GPU.
                assert_eq!(
                    e.code, 100,
                    "unexpected HIP error querying device count: {e}"
                );
            }
        }
    }

    /// Real test: queries actual free/total VRAM, sanity-checks the
    /// relationship (free <= total, total roughly matches this card's real
    /// 24GB), then allocates a real, sizeable buffer and confirms free
    /// VRAM actually DROPS by roughly that amount -- not just that the
    /// call returns success, but that it reflects a real, observable
    /// change in device state.
    #[test]
    fn real_mem_info_reflects_a_real_allocation() {
        if device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let (free_before, total) = mem_info().expect("hipMemGetInfo failed");
        assert!(free_before <= total, "free ({free_before}) must not exceed total ({total})");
        assert!(
            total > 8 * 1024 * 1024 * 1024,
            "total VRAM ({total} bytes) implausibly small for a real discrete GPU"
        );

        let alloc_bytes = 512 * 1024 * 1024usize; // 512MB, big enough to not be noise
        let buf: DeviceBuffer<u8> = DeviceBuffer::alloc(alloc_bytes).expect("hipMalloc failed");
        let (free_after, _) = mem_info().expect("hipMemGetInfo failed (after alloc)");

        let dropped = free_before.saturating_sub(free_after);
        assert!(
            dropped >= alloc_bytes / 2,
            "free VRAM only dropped by {dropped} bytes after a real {alloc_bytes}-byte allocation -- mem_info doesn't seem to reflect real device state"
        );
        drop(buf);
    }

    /// Real round-trip test: allocates real GPU memory, copies real data to
    /// it, copies it back, and checks byte-for-byte equality. This is the
    /// actual claim this whole module makes -- verified, not asserted.
    /// Skips (rather than fails) only if this machine has no visible GPU.
    #[test]
    fn real_alloc_copy_roundtrip() {
        if device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let host_in: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
        let mut buf: DeviceBuffer<f32> =
            DeviceBuffer::alloc(host_in.len()).expect("hipMalloc failed");
        buf.copy_from_host(&host_in)
            .expect("host->device hipMemcpy failed");

        let mut host_out = vec![0.0f32; host_in.len()];
        buf.copy_to_host(&mut host_out)
            .expect("device->host hipMemcpy failed");

        assert_eq!(
            host_in, host_out,
            "round-tripped GPU buffer did not match the original data"
        );
    }

    #[test]
    fn zero_length_alloc_is_rejected() {
        let result: Result<DeviceBuffer<u8>, HipError> = DeviceBuffer::alloc(0);
        assert!(
            result.is_err(),
            "DeviceBuffer::alloc(0) should be rejected, not silently succeed"
        );
    }

    /// §93 feasibility smoke test, isolated from the real decode loop on
    /// purpose (same discipline as §80's original hipcc-compilation smoke
    /// test): captures a real device-to-device `hipMemcpyAsync` (issued on
    /// a real, explicitly-created stream -- capturing the null/legacy
    /// stream directly was tried FIRST and confirmed, not assumed, to fail
    /// with `hipErrorStreamCaptureUnsupported` (code 900) on this system,
    /// matching CUDA/HIP's own documented restriction) into a `hipGraph_t`,
    /// instantiates it, then REPLAYS it twice against the SAME buffers with
    /// DIFFERENT source data each time -- the exact property a real
    /// per-token decode loop needs (fixed buffer addresses, changing
    /// contents, no re-capture between tokens).
    ///
    /// `hipMemcpyAsync` (not `rmsnorm_bf16`) is the operation captured here
    /// on purpose: every `.hip` kernel launcher in this crate currently
    /// hardcodes `stream=0` internally, so none of them can be captured on
    /// an explicit stream without first threading a stream parameter through
    /// all 14 kernel files' FFI + launcher signatures -- a real, separate
    /// effort not worth undertaking before knowing whether graph
    /// capture/replay-with-new-data even works on this system at all.
    /// `hipMemcpyAsync` already takes an explicit stream argument, so it
    /// isolates exactly that question.
    #[test]
    fn real_hip_graph_capture_replay_reflects_new_data_without_recapture() {
        if device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let n = 8usize;
        let mut src_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(n).unwrap();
        let mut dst_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(n).unwrap();
        let bytes = n * std::mem::size_of::<f32>();

        // First real input -- written BEFORE capture. Capture records the
        // copy's parameters (pointers, size, direction); it doesn't need
        // meaningful data present, but writing something real first keeps
        // this test honest about what state the buffer is actually in.
        let x1: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5];
        src_buf.copy_from_host(&x1).unwrap();

        let stream = Stream::create().expect("Stream::create failed");
        begin_capture(&stream).expect("begin_capture failed");
        // SAFETY: `stream.raw()` is a live, capturing stream; the pointers
        // are real, live `hipMalloc` allocations from the `DeviceBuffer`s
        // above, valid for the whole test.
        let code = unsafe {
            ffi::hipMemcpyAsync(
                dst_buf.as_device_ptr_mut() as *mut c_void,
                src_buf.as_device_ptr() as *const c_void,
                bytes,
                ffi::HIP_MEMCPY_DEVICE_TO_DEVICE,
                stream.raw(),
            )
        };
        HipError::from_code(code).expect("hipMemcpyAsync (captured) failed");
        let graph_exec = end_capture(&stream).expect("end_capture failed");

        // Replay 1: real launch of the captured graph, for the FIRST input.
        graph_exec.launch(&stream).expect("graph_exec.launch (replay 1) failed");
        stream.synchronize().expect("stream.synchronize (replay 1) failed");

        let mut out1 = vec![0.0f32; n];
        dst_buf.copy_to_host(&mut out1).unwrap();
        assert_eq!(out1, x1, "replay 1: captured device-to-device copy did not reproduce the source data");

        // THE decisive part: overwrite src_buf with DIFFERENT real data,
        // WITHOUT re-capturing, then replay the SAME graph_exec again.
        let x2: Vec<f32> = vec![-3.0, 1.0, 0.0, 2.0, -1.5, 4.0, -0.25, 0.75];
        src_buf.copy_from_host(&x2).unwrap();

        graph_exec.launch(&stream).expect("graph_exec.launch (replay 2) failed");
        stream.synchronize().expect("stream.synchronize (replay 2) failed");

        let mut out2 = vec![0.0f32; n];
        dst_buf.copy_to_host(&mut out2).unwrap();
        assert_eq!(
            out2, x2,
            "replay 2 (new data, NOT re-captured): graph replay did NOT correctly re-read the live buffer contents"
        );
    }
}
