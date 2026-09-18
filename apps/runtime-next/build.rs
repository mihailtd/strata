//! Links against ROCm's HIP runtime (libamdhip64), found via the standard
//! ROCm install location. No bindgen: the FFI surface in `src/hip.rs` is
//! small and hand-curated on purpose (see that module's doc comment).
//!
//! Also compiles this crate's real hand-written HIP kernels (`src/kernels/
//! *.hip`) via `hipcc` into a shared library and links it. Each `.hip` file
//! is real GPU device code, not a stub -- see `src/kernels/rmsnorm.hip`'s
//! own header comment for what it actually computes and why.

use std::path::Path;
use std::process::Command;

fn main() {
    let rocm_lib = std::env::var("ROCM_LIB_DIR").unwrap_or_else(|_| "/opt/rocm/lib".to_string());
    println!("cargo:rustc-link-search=native={rocm_lib}");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    // hipBLAS: linked for src/blas.rs's real GEMM -- the first feature that
    // links a vendor library instead of hand-writing a .hip kernel (see
    // that module's own doc comment for why).
    println!("cargo:rustc-link-lib=dylib=hipblas");
    // hipBLASLt: §92 performance pass -- src/blaslt.rs's algorithm-tuned
    // GEMM (installed specifically for this: `sudo pacman -S hipblaslt`).
    println!("cargo:rustc-link-lib=dylib=hipblaslt");
    println!("cargo:rerun-if-env-changed=ROCM_LIB_DIR");

    build_hip_kernels();
}

fn build_hip_kernels() {
    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "/opt/rocm/bin/hipcc".to_string());
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set by cargo");
    let kernels_dir = Path::new("src/kernels");

    // Track the directory itself, not just the files enumerated below --
    // otherwise a newly-ADDED .hip file never triggers a rebuild, since
    // cargo only reruns build.rs when a path it previously declared
    // changes, and a brand-new file was never previously declared. (Found
    // this the hard way: adding rope.hip/swiglu.hip/embedding.hip after
    // rmsnorm.hip's first build produced a linker error for undefined
    // symbols, because the stale .so from before those files existed was
    // still what got linked.)
    println!("cargo:rerun-if-changed={}", kernels_dir.display());

    let hip_files: Vec<_> = std::fs::read_dir(kernels_dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", kernels_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("hip"))
        .collect();

    if hip_files.is_empty() {
        return;
    }

    let lib_path = Path::new(&out_dir).join("libruntime_next_kernels.so");
    let mut cmd = Command::new(&hipcc);
    // Without an explicit target, hipcc auto-detects EVERY local GPU
    // agent and builds a fat binary for all of them -- on this real dev
    // machine that's both the real target (`gfx1100`, the RX 7900 XTX
    // this whole engine is written for and the only device the model
    // ever actually loads onto) AND the CPU's incidental integrated
    // graphics (`gfx1036`), which this engine has never used for
    // inference. Real problem this caused: an experimental kernel using
    // RDNA3's `dot8-insts` feature (a real hardware INT8 dot-product
    // instruction -- see docs/DECISIONS.md's real, disclosed writeup of
    // why that experiment was tried and reverted) hard-failed the whole
    // build when compiled for `gfx1036`, which doesn't support it, over
    // a target nothing here runs on. Pin to the one real target instead
    // of guarding every RDNA3-only kernel with `#ifdef`s for a device
    // this crate never uses.
    let offload_arch = std::env::var("RUNTIME_NEXT_OFFLOAD_ARCH").unwrap_or_else(|_| "gfx1100".to_string());
    cmd.arg(format!("--offload-arch={offload_arch}"));
    println!("cargo:rerun-if-env-changed=RUNTIME_NEXT_OFFLOAD_ARCH");
    cmd.arg("-fPIC").arg("-shared").arg("-O2");
    for f in &hip_files {
        println!("cargo:rerun-if-changed={}", f.display());
        cmd.arg(f);
    }
    cmd.arg("-o").arg(&lib_path);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!("failed to invoke {hipcc}: {e} (set HIPCC to override the path)")
    });
    if !status.success() {
        panic!("hipcc failed to compile {:?} (exit: {status})", hip_files);
    }

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=dylib=runtime_next_kernels");
    // Also needed at runtime (not just link time) unless installed into the
    // default loader search path -- cargo test/run both need to find it.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{out_dir}");
}
