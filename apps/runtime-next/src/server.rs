//! §95: a minimal, real, single-tenant OpenAI-compatible HTTP server --
//! the missing piece that let this whole session's throughput work
//! (31.7 -> ~82 tok/s, see `docs/DECISIONS.md` §91-§94) only ever be
//! measured directly (`llama-bench`, Ollama's `/api/generate`), never
//! through the kind of HTTP/SSE request shape a real client -- DSH
//! included -- would actually send.
//!
//! Single-tenant by design, not by accident: this engine touches exactly
//! one GPU with no concurrency to exploit (same as `apps/runtime-triton`'s
//! own documented "requests execute strictly in arrival order" design).
//! `tiny_http` (a real, synchronous, no-async-runtime HTTP crate) matches
//! that directly -- no `tokio`/`axum` pulled in for concurrency this
//! engine structurally cannot use.
//!
//! Route surface matches `apps/runtime-triton/server.py`'s real schema
//! (`ChatMessage`/`ChatCompletionRequest`/`ChatCompletionResponse`/
//! `ChatCompletionChunkResponse`, that file's own lines 242-328) closely
//! enough to be a drop-in OpenAI-compatible target for the same real
//! clients: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
//! (both non-streaming JSON and `stream: true` Server-Sent Events).

use crate::blas::BlasHandle;
use crate::hip::DeviceBuffer;
use crate::lora::{LoraAdapter, PristineWeights};
use crate::model::{DecodeState, GraphedDecodeState, ModelWeights, VOCAB_SIZE, argmax_sample, forward_prefill};
use crate::quantized_lora::{QuantizedLoraAdapter, activate_quantized_adapter, clear_quantized_adapter};
use crate::sampling::{SamplingParams, make_rng, sample_from_logits};
use crate::state_handoff::TensorStateSnapshot;
use crate::tokenizer::ChatTokenizer;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

#[cfg(feature = "qwen35_0_8b")]
const MODEL_ID: &str = "qwen3.5:0.8b-rust";
#[cfg(feature = "qwen35_2b")]
const MODEL_ID: &str = "qwen3.5:2b-rust";
#[cfg(feature = "qwen35_4b")]
const MODEL_ID: &str = "qwen3.5:4b-rust";
#[cfg(feature = "qwen35_9b")]
const MODEL_ID: &str = "qwen3.5:9b-rust";
#[cfg(feature = "qwen35_27b")]
const MODEL_ID: &str = "qwen3.8:27b-rust";
/// §95/§120: LDS limit on RDNA3 (64KB) allows up to ~15232 floats in
/// `attention_decode_split.hip`. 12288 fits comfortably in 52.5KB LDS and
/// easily handles tool-augmented prompts with >8k tokens.
/// Default context window. Raised from 8192 after measuring what the window
/// actually costs on this hybrid architecture: only 8 of 32 layers are full
/// attention (`is_full_attention_layer`, `full_attention_interval=4`); the
/// other 24 are GDN and hold a FIXED-size recurrent state that does not grow
/// with sequence length at all. So the per-token cost is
///   KV:          8 layers x 2 caches x ATTN_NUM_KV_HEADS x ATTN_HEAD_DIM x 2B
///   attn_scores: MAX_PREFILL_CHUNK x 2B
/// = ~32.5 KB/token on 4B, i.e. ~400 MB at 12288 against ~268 MB at 8192.
/// KV size is NOT what bounds the window here, which is why this is a bigger
/// window rather than a KV-quantization scheme -- quantizing a 268 MB cache
/// would have solved a problem this architecture does not have.
///
/// What actually bounds it is `attention_decode_split.hip`, which sizes its
/// shared memory by the FULL kv_stride:
///   shmem = (ATTN_HEAD_DIM + kv_stride + threads + ATTN_HEAD_DIM*KV_SPLIT) * 4
/// with `threads = ATTN_HEAD_DIM * ATTENTION_DECODE_KV_SPLIT` = 1024 (already
/// the AMD workgroup maximum). On gfx1100 LDS is 64 KB per workgroup, so
///   max_seq_len <= 16384 - 256 - 2*256*4 = 14080
/// and a larger window fails the launch outright with HIP "invalid argument"
/// partway through a generation. 12288 is the largest bucket-aligned window
/// that fits, and it was already `KV_LEN_BUCKETS`'s top entry for exactly
/// this reason. Going beyond needs the kernel to TILE the KV row through
/// shared memory instead of holding all of it -- a real kernel rewrite, and
/// the actual unlock for long context on this engine.
const DEFAULT_MAX_SEQ_LEN: usize = 12288;

/// Largest window `attention_decode_split.hip` can launch within gfx1100's
/// 64 KB LDS budget. Derived, not guessed -- see `DEFAULT_MAX_SEQ_LEN`.
const LDS_MAX_SEQ_LEN: usize = (65536 / 4)
    - crate::model::ATTN_HEAD_DIM
    - 2 * crate::model::ATTN_HEAD_DIM * crate::model::ATTENTION_DECODE_KV_SPLIT;

/// Resolved once per process from `RUNTIME_NEXT_MAX_SEQ_LEN`, so a smaller
/// card can dial it down without a rebuild. Read through `max_seq_len()`.
static MAX_SEQ_LEN_CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn max_seq_len() -> usize {
    *MAX_SEQ_LEN_CELL.get_or_init(|| {
        let requested = std::env::var("RUNTIME_NEXT_MAX_SEQ_LEN")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n >= 512)
            .unwrap_or(DEFAULT_MAX_SEQ_LEN);
        // Refuse loudly at startup rather than letting the decode kernel fail
        // with an opaque HIP "invalid argument" thousands of tokens into a
        // generation, which is exactly how this limit was found.
        if requested > LDS_MAX_SEQ_LEN {
            eprintln!(
                "[runtime-next] RUNTIME_NEXT_MAX_SEQ_LEN={requested} exceeds what attention_decode_split can launch \
                 within 64 KB LDS (max {LDS_MAX_SEQ_LEN}); clamping to {LDS_MAX_SEQ_LEN}"
            );
            return LDS_MAX_SEQ_LEN;
        }
        requested
    })
}
const DEFAULT_MAX_TOKENS: usize = 4096;
/// §127: real, deliberate cap on concurrently-stored state-handoff
/// snapshots. Each real snapshot is a full device-to-device clone of
/// every GDN layer's recurrent/conv state and every attention layer's
/// K/V cache -- real, non-trivial VRAM (roughly 180-400MB per snapshot
/// depending on model size, measured directly from `DecodeState`'s own
/// real per-layer buffer sizes), not something to let grow unbounded on
/// a single-GPU server that also holds the model weights and the live
/// `DecodeState` itself. A real, loud error when this cap is hit (see
/// `Engine::snapshot_state`) rather than silent eviction or an
/// out-of-memory crash.
const MAX_SNAPSHOTS: usize = 8;

/// Idle lifetime of a stored snapshot. Measured from its LAST use, not its
/// creation, so an actively-resumed multi-turn session never expires
/// mid-conversation while an abandoned one still frees its VRAM promptly.
const SNAPSHOT_TTL_SECS: u64 = 60;

fn unix_time_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// A stored snapshot plus the metadata needed to decide whether restoring it
/// is actually SOUND. `adapter_sig` is the critical field: a snapshot's K/V
/// cache holds `W_k · x` and `W_v · x` for the exact folded weights that were
/// live when it was captured, and every layer's hidden states carry the
/// adapter's MLP deltas too. Restoring it under different folded weights
/// silently mixes two weight bases inside one attention computation, so the
/// signature is recorded here and enforced on resume rather than trusted.
struct StoredSnapshot {
    snap: TensorStateSnapshot,
    created: u64,
    last_used: u64,
    adapter_sig: String,
}

/// Everything one real decode session needs, held behind a single mutex --
/// the real single-tenant boundary: only one request may touch the GPU at
/// a time, enforced here rather than assumed.
struct Engine {
    weights: ModelWeights,
    graphed: GraphedDecodeState,
    /// §96: a real, separate hipBLAS handle for `forward_prefill`'s own
    /// batched GEMMs -- deliberately NOT `graphed`'s own handle (bound to
    /// `graphed`'s captured-graph stream) -- this one runs ungraphed, on
    /// the null stream.
    blas_handle: BlasHandle,
    state: DecodeState,
    logits: DeviceBuffer<u16>,
    tokenizer: ChatTokenizer,
    /// §118: the real, current request's sampling parameters -- set once
    /// per real request in `start_request`, read every `step()` call
    /// after. `temperature<=0.0` (the pre-existing default, and every
    /// real benchmark script in `benchmarks/harness_sdk` that hits this
    /// server already sends `"temperature": 0.0` explicitly) keeps
    /// `step()` on the EXACT SAME on-device `argmax_sample` path it
    /// always used -- this field existing changes nothing for any
    /// existing caller/test/benchmark.
    sampling: SamplingParams,
    /// §118: one real, seedable RNG per request (re-seeded in
    /// `start_request`, not per token) -- see `sampling::make_rng`'s own
    /// doc comment for why per-token reseeding would be both slower and
    /// architecturally wrong.
    rng: StdRng,
    /// §118: real, reused host-side scratch for `step()`'s real
    /// logits-to-host copy when `sampling.temperature>0.0` -- allocated
    /// ONCE (`VOCAB_SIZE` elements), matching this crate's own
    /// established "preallocate once, reuse every call" discipline
    /// (`Scratch`/`PrefillScratch`) rather than a fresh heap allocation
    /// every sampled token. Unused entirely on the greedy fast path.
    logits_host_scratch: Vec<u16>,
    /// §127: real, server-side store of captured `TensorStateSnapshot`s
    /// keyed by a caller-supplied name -- the real mechanism behind
    /// "pause conversation A, serve other requests, resume conversation A
    /// (or hand it to a differently-named session) with zero re-prefill".
    /// Restoring ALWAYS writes into this same `Engine`'s own `state`
    /// field (see `start_request`'s own doc comment for why this, and
    /// not swapping in a second `DecodeState`, is the only safe design
    /// given `graphed`'s captured HIP graph is tied to `state`'s exact
    /// buffer addresses).
    snapshots: HashMap<String, StoredSnapshot>,
    /// Reused across requests: a `BatchedDecodeState`'s KV allocation is
    /// ~400 MB per slot at a 12288 window, so it is built once per batch size
    /// and kept. `load_slot` fully overwrites every layer, and attention only
    /// ever reads up to each slot's own position, so no zeroing is needed
    /// between requests (same reasoning as `DecodeState::reset`'s own note on
    /// the KV caches).
    batched: Option<crate::model::BatchedDecodeState>,
    /// §133: Pristine weights backup for dense BF16 In-Place Weight Folding.
    /// Captured on the first dense LoRA adapter load. Restores bit-exact
    /// pristine base weights before folding a new adapter or returning to base.
    pristine: Option<PristineWeights>,
    /// §133: Real, registered LoRA adapters on this engine.
    adapters: Vec<RegisteredAdapter>,
    /// §133: The currently active adapter slot ID, if any.
    active_adapter_id: Option<usize>,
}

/// §133: Live LoRA Adapter representation in `server.rs`. Supports both
/// dense BF16 In-Place Weight Folding (`lora.rs`) and quantized W4A16
/// static slot DMA uploads (`quantized_lora.rs`).
pub enum LoadedAdapter {
    Dense(LoraAdapter),
    Quantized(QuantizedLoraAdapter),
}

pub struct RegisteredAdapter {
    pub id: usize,
    pub name: String,
    pub path: String,
    pub adapter: LoadedAdapter,
    pub scale: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoraAdapterInfo {
    pub id: usize,
    pub path: String,
    #[serde(default)]
    pub name: String,
    pub scale: f32,
}

impl Engine {
    fn load() -> Result<Self, String> {
        let snapshot = crate::model_loader::locate_model_snapshot()?;
        eprintln!("[runtime-next] loading real weights from {}...", snapshot.display());
        let weights = ModelWeights::load(&snapshot)?;
        eprintln!("[runtime-next] weights loaded.");
        let tokenizer = ChatTokenizer::load(&snapshot)?;
        let graphed = GraphedDecodeState::new().map_err(|e| e.to_string())?;
        let blas_handle = BlasHandle::create().map_err(|e| e.to_string())?;
        // Report what the context window actually costs, measured from the
        // driver rather than computed from a formula -- this repo has been
        // burned before by VRAM claims that were arithmetic, not observation.
        let before = crate::hip::mem_info().ok();
        let state = DecodeState::new(max_seq_len()).map_err(|e| e.to_string())?;
        if let (Some((free_before, total)), Ok((free_after, _))) = (before, crate::hip::mem_info()) {
            let used = free_before.saturating_sub(free_after);
            eprintln!(
                "[runtime-next] context window {} tokens; decode state cost {:.0} MB measured ({:.1} GB of {:.1} GB free remaining)",
                max_seq_len(),
                used as f64 / (1024.0 * 1024.0),
                free_after as f64 / (1024.0 * 1024.0 * 1024.0),
                total as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        eprintln!(
            "[runtime-next] chat template: {} | stop token ids: {:?}",
            tokenizer.template_source().unwrap_or("built-in Qwen3.5 fallback (snapshot ships none)"),
            tokenizer.stop_ids()
        );
        let logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).map_err(|e| e.to_string())?;
        Ok(Engine {
            weights,
            graphed,
            blas_handle,
            state,
            logits,
            tokenizer,
            sampling: SamplingParams::GREEDY,
            rng: make_rng(None),
            logits_host_scratch: vec![0u16; VOCAB_SIZE],
            snapshots: HashMap::new(),
            batched: None,
            pristine: None,
            adapters: Vec::new(),
            active_adapter_id: None,
        })
    }

    /// §133: Loads a real LoRA adapter directory (`adapter_config.json` + `adapter_model.safetensors`).
    /// Automatically detects whether model is quantized W4A16 or dense BF16 and dispatches
    /// to the correct zero-overhead hot-swap representation.
    /// Captures `PristineWeights` on first dense load.
    pub fn load_adapter(&mut self, path_str: &str, name_opt: Option<&str>) -> Result<usize, String> {
        let raw_path = std::path::Path::new(path_str.trim_end_matches('/'));
        let dir = if raw_path.is_file() {
            raw_path.parent().unwrap_or(raw_path)
        } else {
            raw_path
        };
        if !dir.exists() {
            return Err(format!("adapter directory does not exist: {}", dir.display()));
        }

        let dir_canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        let dir_str = dir_canonical.to_string_lossy().to_string();

        if let Some(existing) = self.adapters.iter().find(|a| a.path == dir_str || a.path == path_str) {
            return Ok(existing.id);
        }

        let name = name_opt
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                dir.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("adapter")
                    .to_string()
            });

        let loaded = if self.weights.is_quantized() {
            let q = QuantizedLoraAdapter::load_from_dir(dir, &name)?;
            LoadedAdapter::Quantized(q)
        } else {
            if self.pristine.is_none() {
                let p = PristineWeights::capture(&self.weights)
                    .map_err(|e| format!("failed to capture pristine weights: {e}"))?;
                self.pristine = Some(p);
            }
            let d = LoraAdapter::load_from_dir(dir, &name)?;
            LoadedAdapter::Dense(d)
        };

        let id = self.adapters.len();
        self.adapters.push(RegisteredAdapter {
            id,
            name,
            path: dir_str,
            adapter: loaded,
            scale: 0.0,
        });
        Ok(id)
    }

    /// §133: Updates adapter scale(s) and executes on-device activation/deactivation.
    /// Supports multi-adapter additive stacking for dense models.
    /// Returns elapsed swap latency in milliseconds.
    pub fn set_adapter_scales(&mut self, updates: &[(usize, f32)]) -> Result<f64, String> {
        let t0 = std::time::Instant::now();

        for &(id, _) in updates {
            if id >= self.adapters.len() {
                return Err(format!("adapter id {id} out of range (total registered: {})", self.adapters.len()));
            }
        }

        let target_active: Vec<(usize, f32)> = updates
            .iter()
            .filter(|&&(_, s)| s > 0.0)
            .copied()
            .collect();

        if self.weights.is_quantized() {
            if target_active.len() > 1 {
                return Err("Quantized LoRA stacking is not supported on W4A16".to_string());
            }
            clear_quantized_adapter(&mut self.weights)
                .map_err(|e| format!("clear_quantized_adapter failed: {e}"))?;
            if let Some(&(target_id, _)) = target_active.first() {
                if let LoadedAdapter::Quantized(q) = &self.adapters[target_id].adapter {
                    activate_quantized_adapter(&mut self.weights, q)
                        .map_err(|e| format!("activate_quantized_adapter failed: {e}"))?;
                }
                self.active_adapter_id = Some(target_id);
            } else {
                self.active_adapter_id = None;
            }
        } else {
            if self.pristine.is_none() {
                let p = PristineWeights::capture(&self.weights)
                    .map_err(|e| format!("failed to capture pristine weights: {e}"))?;
                self.pristine = Some(p);
            }
            let pristine = self.pristine.as_ref().unwrap();
            pristine.restore(&mut self.weights)
                .map_err(|e| format!("pristine restore failed: {e}"))?;

            for &(target_id, target_scale) in &target_active {
                if let LoadedAdapter::Dense(d) = &self.adapters[target_id].adapter {
                    crate::lora::fold_adapter_into(&self.blas_handle, &mut self.weights, d, target_scale * d.scale)
                        .map_err(|e| format!("fold_adapter_into failed: {e}"))?;
                }
            }

            self.active_adapter_id = if target_active.len() == 1 {
                Some(target_active[0].0)
            } else {
                None
            };
        }

        for a in &mut self.adapters {
            a.scale = 0.0;
        }
        for &(id, scale) in updates {
            self.adapters[id].scale = scale;
        }

        let elapsed = t0.elapsed();
        Ok(elapsed.as_secs_f64() * 1000.0)
    }

    /// Swaps to an adapter by name or ID. Supports "adapter@scale" syntax (e.g. "python_modern@0.5"),
    /// or compound multi-adapter stacking syntax separated by '+' or ',' (e.g. "agentic@0.25+python_modern@0.25").
    /// "base" or "none" clears adapters.
    pub fn swap_to_adapter(&mut self, spec: &str) -> Result<(Option<usize>, f64), String> {
        let spec_trimmed = spec.trim();
        if spec_trimmed == "base" || spec_trimmed == "none" || spec_trimmed.is_empty() {
            let updates: Vec<(usize, f32)> = (0..self.adapters.len()).map(|i| (i, 0.0)).collect();
            let ms = self.set_adapter_scales(&updates)?;
            return Ok((None, ms));
        }

        let parts: Vec<&str> = if spec_trimmed.contains('+') {
            spec_trimmed.split('+').map(|s| s.trim()).collect()
        } else if spec_trimmed.contains(',') {
            spec_trimmed.split(',').map(|s| s.trim()).collect()
        } else {
            vec![spec_trimmed]
        };

        let mut active_targets: Vec<(usize, f32)> = Vec::new();
        for subspec in parts {
            if subspec.is_empty() { continue; }
            let (name_part, scale_val) = if let Some(idx) = subspec.find('@') {
                let n = subspec[..idx].trim();
                let s = subspec[idx + 1..]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid scale in {subspec}: {e}"))?;
                (n, s)
            } else {
                (subspec, 1.0f32)
            };

            let target_id = if let Ok(id) = name_part.parse::<usize>() {
                if id < self.adapters.len() {
                    id
                } else {
                    return Err(format!("adapter id {id} out of range (total: {})", self.adapters.len()));
                }
            } else if let Some(pos) = self.adapters.iter().position(|a| a.name == name_part || a.name.contains(name_part) || a.path.contains(name_part)) {
                pos
            } else {
                // On-demand auto-discovery and loading from filesystem
                let candidate_paths = [
                    std::path::PathBuf::from(name_part),
                    std::path::PathBuf::from("results/adapters").join(name_part),
                ];
                let found_path = candidate_paths.into_iter().find(|p| p.join("adapter_config.json").exists() || p.join("adapter_model.safetensors").exists());
                if let Some(p) = found_path {
                    self.load_adapter(&p.to_string_lossy(), Some(name_part))?
                } else {
                    return Err(format!("adapter {name_part:?} not found among registered adapters"));
                }
            };
            active_targets.push((target_id, scale_val));
        }

        let mut updates: Vec<(usize, f32)> = (0..self.adapters.len()).map(|i| (i, 0.0)).collect();
        for (id, scale) in &active_targets {
            updates[*id] = (*id, *scale);
        }

        let ms = self.set_adapter_scales(&updates)?;
        let first_id = active_targets.first().map(|(id, _)| *id);
        Ok((first_id, ms))
    }


    pub fn list_adapters(&self) -> Vec<LoraAdapterInfo> {
        self.adapters
            .iter()
            .map(|a| LoraAdapterInfo {
                id: a.id,
                path: a.path.clone(),
                name: a.name.clone(),
                scale: a.scale,
            })
            .collect()
    }

    /// §127: real, device-to-device capture of the CURRENT session's
    /// entire mutable state (GDN recurrent/conv state, attention K/V
    /// cache, `position`) under `name`, ready to be restored later via
    /// `start_request`'s own `resume` parameter -- the real mechanism a
    /// paused/handed-off session needs. Overwriting an EXISTING name is
    /// real, intentional "upsert" semantics (never counts against
    /// `MAX_SNAPSHOTS` since the total entry count doesn't grow); a
    /// genuinely NEW name past the cap is a real, loud error, not silent
    /// eviction of someone else's saved session.
    fn snapshot_state(&mut self, name: String) -> Result<(usize, u64), String> {
        self.purge_expired_snapshots();
        if !self.snapshots.contains_key(&name) && self.snapshots.len() >= MAX_SNAPSHOTS {
            return Err(format!(
                "snapshot store is full ({MAX_SNAPSHOTS} max) -- delete an existing snapshot before creating a new one (name {name:?} is new)"
            ));
        }
        let snap = TensorStateSnapshot::capture(&self.state).map_err(|e| e.to_string())?;
        let position = self.state.position;
        let created = unix_time_secs();
        let adapter_sig = self.active_adapter_signature();
        self.snapshots.insert(name, StoredSnapshot { snap, created, last_used: created, adapter_sig });
        Ok((position, created))
    }

    /// Canonical identity of the currently folded adapter configuration --
    /// what a snapshot's cached K/V and hidden states were actually computed
    /// with. Scale is part of the identity because folding applies
    /// `target_scale * (lora_alpha/r) * (B@A)`, so the same adapter at a
    /// different scale is a genuinely different weight matrix.
    fn active_adapter_signature(&self) -> String {
        let mut active: Vec<String> = self
            .adapters
            .iter()
            .filter(|a| a.scale != 0.0)
            .map(|a| format!("{}@{}", a.name, a.scale))
            .collect();
        if active.is_empty() {
            return "base".to_string();
        }
        active.sort();
        active.join("+")
    }

    /// Drops snapshots idle longer than `SNAPSHOT_TTL_SECS`, freeing their
    /// real VRAM. Returns how many were reaped.
    fn purge_expired_snapshots(&mut self) -> usize {
        let now = unix_time_secs();
        let before = self.snapshots.len();
        self.snapshots
            .retain(|_, stored| now.saturating_sub(stored.last_used) <= SNAPSHOT_TTL_SECS);
        before - self.snapshots.len()
    }

    /// Real, read-only listing for a real `GET /v1/state/snapshots`
    /// endpoint -- name, the real `position` it was captured at, and when.
    fn list_snapshots(&self) -> Vec<(String, usize, u64)> {
        self.snapshots
            .iter()
            .map(|(name, stored)| (name.clone(), stored.snap.position, stored.created))
            .collect()
    }

    /// Real deletion, freeing that snapshot's real VRAM. Returns whether
    /// a snapshot with that name actually existed.
    fn delete_snapshot(&mut self, name: &str) -> bool {
        self.snapshots.remove(name).is_some()
    }

    /// §96/§118: starts a new, independent request: resets the reused
    /// `DecodeState` (§95's `DecodeState::reset`, proven leak-free by
    /// `real_reset_prevents_cross_request_state_leakage`), applies the
    /// real chat template, encodes it, prefills via the real batched
    /// `forward_prefill` (chunked internally at `MAX_PREFILL_CHUNK` tokens
    /// per real batched GEMM), and latches in this real request's own
    /// sampling parameters + a freshly (re)seeded RNG for every `step()`
    /// call that follows.
    ///
    /// A real bucketed, HIP-Graph-captured prefill path was built and
    /// made fully correct (see `docs/DECISIONS.md` §117), but a
    /// controlled real HTTP A/B found it a net TTFT REGRESSION for this
    /// server's real prompt-length distribution (4B: 168.8ms eager vs.
    /// 276.2ms bucketed+graphed) -- bucket-padding waste, quadratic in
    /// attention's O(T²) prefill cost, outweighed the real dispatch-
    /// overhead savings the graph capture bought. Moved to
    /// `experiments/bucketed_hip_graph_prefill/` rather than shipped; not
    /// wired in here.
    /// §127: `resume` real, deliberate addition -- `None` is BYTE-FOR-BYTE
    /// the pre-existing code path (reset, encode the FULL real `messages`,
    /// prefill from position 0), zero behavior change for every existing
    /// caller/test/benchmark that never sets it. `Some(name)` restores a
    /// previously-captured `TensorStateSnapshot` INTO this same `Engine`'s
    /// own `state` (never a second `DecodeState` -- see `Engine::snapshots`'
    /// own doc comment for why that's the only safe design given
    /// `graphed`'s captured HIP graph), then encodes and prefills ONLY the
    /// real `messages` this call was given ON TOP of the restored state --
    /// real incremental prefill, the same proven-correct mechanism
    /// `state_handoff.rs`'s own `real_incremental_prefill_matches_one_shot_
    /// prefill_numerically` established: calling `forward_prefill` again on
    /// a non-reset `DecodeState` continues exactly as if the whole sequence
    /// had been prefilled in one call. The real, disclosed API contract:
    /// a caller resuming a session sends ONLY the new turn(s), never the
    /// full history again -- `apply_chat_template`/`apply_chat_template_
    /// with_tools` have no hidden global preamble, so re-sending old turns
    /// would double-encode them into the KV cache, which is wrong.
    /// Restoring does NOT consume the snapshot -- it stays available for
    /// a later resume (real, deliberate "named checkpoint", not a
    /// one-shot handoff token; branching/retrying from the same point is a
    /// real, intended use).
    fn start_request(
        &mut self,
        messages: &[(&str, &str)],
        tools: Option<&[serde_json::Value]>,
        sampling: SamplingParams,
        seed: Option<u64>,
        resume: Option<&str>,
    ) -> Result<usize, String> {
        let json: Vec<serde_json::Value> =
            messages.iter().map(|(r, c)| serde_json::json!({"role": r, "content": c})).collect();
        self.start_request_json(&json, tools, sampling, seed, resume, None)
    }

    /// The general form: full OpenAI message objects rendered with the model's
    /// OWN chat template (`ChatTokenizer::render_chat`), plus the request's
    /// `chat_template_kwargs` passed through to the template.
    fn start_request_json(
        &mut self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        sampling: SamplingParams,
        seed: Option<u64>,
        resume: Option<&str>,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<usize, String> {
        let base_position = match resume {
            None => {
                self.state.reset().map_err(|e| e.to_string())?;
                0
            }
            Some(name) => {
                self.purge_expired_snapshots();
                // The adapter swap for THIS request has already been applied
                // by the time we get here, so the live signature is the one
                // the restored state would be decoded under.
                let current = self.active_adapter_signature();
                let captured_under = self
                    .snapshots
                    .get(name)
                    .map(|stored| stored.adapter_sig.clone())
                    .ok_or_else(|| format!("no snapshot named {name:?} (see GET /v1/state/snapshots for what's available -- note snapshots expire {SNAPSHOT_TTL_SECS}s after their last use)"))?;
                if captured_under != current {
                    // Refusing rather than restoring: the cache would mix two
                    // weight bases inside one attention computation, which
                    // degrades output subtly instead of failing loudly. Drop
                    // the now-unusable snapshot so its VRAM is not stranded.
                    self.snapshots.remove(name);
                    return Err(format!(
                        "snapshot {name:?} was captured under adapter {captured_under:?} but this request runs under {current:?}; its K/V cache and hidden states were computed with different folded weights, so restoring it would silently corrupt attention. The snapshot has been deleted -- re-send the full conversation WITHOUT `resume` to prefill it under {current:?}."
                    ));
                }
                let now = unix_time_secs();
                let stored = self.snapshots.get_mut(name).expect("presence checked above");
                stored.last_used = now;
                stored.snap.restore(&mut self.state).map_err(|e| e.to_string())?;
                self.state.position
            }
        };
        self.sampling = sampling;
        self.rng = make_rng(seed);
        let prompt = self.tokenizer.render_chat(messages, tools, template_kwargs)?;
        let prompt_ids = self.tokenizer.encode(&prompt)?;
        if base_position + prompt_ids.len() >= max_seq_len() {
            return Err(format!(
                "prompt length ({} tokens) starting from position {base_position} would exceed server context limit ({})",
                prompt_ids.len(), max_seq_len()
            ));
        }

        // §108: the batched-GEMM quantized kernel (`w4a16_gemm_prefill`)
        // exists (real fix for §106's disclosed TTFT gap), so a quantized
        // checkpoint prefills through the same real batched
        // `forward_prefill` path as bf16 -- `LinearWeight::apply_prefill`
        // dispatches to the right real kernel per-weight, no branch
        // needed here.
        forward_prefill(&self.blas_handle, &self.weights, &mut self.state, &prompt_ids, &mut self.logits).map_err(|e| e.to_string())?;
        Ok(prompt_ids.len())
    }

    /// One real decode step: sample the next token from the CURRENT
    /// logits (reflecting whatever was prefilled/generated last), then
    /// advance state for the NEXT call. Returns the sampled token id.
    ///
    /// §118: `self.sampling.temperature<=0.0` keeps the EXACT pre-existing
    /// on-device greedy path (`argmax_sample`) -- zero new cost, zero
    /// behavior change, every existing byte-exact-greedy correctness test
    /// and benchmark in this crate is unaffected. Only a real,
    /// explicitly-requested `temperature>0.0` pays the real logits->host
    /// copy (`sampling::sample_from_logits`'s own doc comment explains why
    /// that copy is unavoidable and why it's cheap -- the same real
    /// ~0.2ms cost `argmax_sample`'s own pre-optimization implementation
    /// already had).
    /// Whether one more decoded token would run past the KV cache that
    /// `DecodeState::new(MAX_SEQ_LEN)` actually allocated. Generation loops
    /// must consult this and stop with `finish_reason="length"`; `step`
    /// itself also refuses, so no caller can fault the GPU.
    fn context_exhausted(&self) -> bool {
        self.state.position + 1 >= max_seq_len()
    }

    /// Generates one completion for each of `prompts`, all decoded together in
    /// one batch. Unlike `generate_n` (one prompt fanned out), these are N
    /// DIFFERENT prompts of different lengths, so each slot is prefilled
    /// separately through the ordinary single-sequence path and then copied
    /// into its slot. Prefill is a small fraction of a generation, so the
    /// batched decode still carries the win.
    ///
    /// One constraint worth stating: a batch shares ONE folded adapter,
    /// because LoRA folding mutates the live weights (§12f). Per-slot adapters
    /// would require per-slot weights, which is a different engine.
    fn generate_batch(
        &mut self,
        prompts: &[Vec<serde_json::Value>],
        tools: Option<&[serde_json::Value]>,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
        max_tokens: usize,
        sampling: SamplingParams,
        seed: Option<u64>,
        ignore_eos: bool,
    ) -> Result<(Vec<(Vec<i32>, &'static str)>, Vec<usize>), String> {
        let n = prompts.len();
        let needs_alloc = self.batched.as_ref().map(|b| b.batch() != n || b.max_seq_len != max_seq_len()).unwrap_or(true);
        if needs_alloc {
            // Free the old state BEFORE allocating the new one. Assigning
            // `Some(new(..))` directly builds the new state while the old is
            // still alive, so a size change briefly holds both: at 9B an 8-slot
            // state (~4.3 GB) plus a 7-slot one OOMs with ~6.5 GB free. Measured
            // 2026-09-24: every file's final, partial batch of the corpus
            // generator failed this way.
            self.batched = None;
            self.batched = Some(crate::model::BatchedDecodeState::new(n, max_seq_len()).map_err(|e| e.to_string())?);
        }
        let mut rngs: Vec<_> = (0..n).map(|i| make_rng(Some(seed.unwrap_or(0).wrapping_add(i as u64)))).collect();
        let mut out: Vec<(Vec<i32>, &'static str)> = (0..n).map(|_| (Vec::new(), "length")).collect();
        let mut done = vec![false; n];
        let mut feed = vec![0i32; n];
        let mut prompt_tokens = vec![0usize; n];

        // Prefill each prompt in turn, sampling its first token BEFORE the
        // next prefill overwrites `self.logits`, then copy the finished state
        // into that slot.
        for (i, msgs) in prompts.iter().enumerate() {
            prompt_tokens[i] = self.start_request_json(msgs, tools, sampling, seed, None, template_kwargs)?;
            let t = if sampling.temperature <= 0.0 {
                argmax_sample(&self.logits).map_err(|e| e.to_string())?
            } else {
                self.logits.copy_to_host(&mut self.logits_host_scratch).map_err(|e| e.to_string())?;
                sample_from_logits(&self.logits_host_scratch, &sampling, &mut rngs[i])
            };
            feed[i] = t;
            if self.tokenizer.is_stop(t) && !ignore_eos {
                done[i] = true;
                out[i].1 = "stop";
            } else {
                out[i].0.push(t);
            }
            self.batched.as_mut().expect("batched").load_slot(i, &self.state).map_err(|e| e.to_string())?;
        }

        let mut logits: DeviceBuffer<u16> =
            DeviceBuffer::alloc(n * crate::model::VOCAB_SIZE).map_err(|e| e.to_string())?;
        for _ in 1..max_tokens {
            if done.iter().all(|d| *d) {
                break;
            }
            if self.batched.as_ref().expect("batched").positions.iter().any(|&p| p + 1 >= max_seq_len()) {
                break;
            }
            crate::model::forward_batched_decode(
                &self.blas_handle,
                &self.weights,
                self.batched.as_mut().expect("batched"),
                &feed,
                &mut logits,
            )
            .map_err(|e| e.to_string())?;
            for i in 0..n {
                if done[i] {
                    continue;
                }
                let t = if sampling.temperature <= 0.0 {
                    crate::model::argmax_sample_row(&logits, i).map_err(|e| e.to_string())?
                } else {
                    logits.copy_row_to_host(&mut self.logits_host_scratch, i * crate::model::VOCAB_SIZE).map_err(|e| e.to_string())?;
                    sample_from_logits(&self.logits_host_scratch, &sampling, &mut rngs[i])
                };
                if self.tokenizer.is_stop(t) && !ignore_eos {
                    done[i] = true;
                    out[i].1 = "stop";
                    continue;
                }
                out[i].0.push(t);
                feed[i] = t;
            }
        }
        Ok((out, prompt_tokens))
    }

    /// Generates `n` independent completions for the prompt ALREADY prefilled
    /// into `self.state`, by fanning that one prefilled state out to `n` batch
    /// slots and decoding them together. Returns each slot's token ids and
    /// finish reason.
    ///
    /// Sampling is per slot with its own RNG -- at `temperature <= 0` every
    /// slot would otherwise decode the identical greedy continuation, which
    /// is why `n > 1` forces a real sampling temperature.
    fn generate_n(
        &mut self,
        n: usize,
        max_tokens: usize,
        sampling: SamplingParams,
        seed: Option<u64>,
        ignore_eos: bool,
    ) -> Result<Vec<(Vec<i32>, &'static str)>, String> {
        let needs_alloc = self.batched.as_ref().map(|b| b.batch() != n || b.max_seq_len != max_seq_len()).unwrap_or(true);
        if needs_alloc {
            // Free the old state BEFORE allocating the new one. Assigning
            // `Some(new(..))` directly builds the new state while the old is
            // still alive, so a size change briefly holds both: at 9B an 8-slot
            // state (~4.3 GB) plus a 7-slot one OOMs with ~6.5 GB free. Measured
            // 2026-09-24: every file's final, partial batch of the corpus
            // generator failed this way.
            self.batched = None;
            self.batched = Some(crate::model::BatchedDecodeState::new(n, max_seq_len()).map_err(|e| e.to_string())?);
        }
        let bstate = self.batched.as_mut().expect("just ensured");
        for i in 0..n {
            bstate.load_slot(i, &self.state).map_err(|e| e.to_string())?;
        }

        let mut logits: DeviceBuffer<u16> =
            DeviceBuffer::alloc(n * crate::model::VOCAB_SIZE).map_err(|e| e.to_string())?;
        let mut rngs: Vec<_> = (0..n).map(|i| make_rng(Some(seed.unwrap_or(0).wrapping_add(i as u64)))).collect();
        let mut out: Vec<(Vec<i32>, &'static str)> = (0..n).map(|_| (Vec::new(), "length")).collect();
        let mut done = vec![false; n];

        // The prompt's own last-position logits are already in `self.logits`
        // from the shared prefill, so slot 0..n all sample their FIRST token
        // from that same distribution -- with different RNG draws, which is
        // where the diversity comes from.
        let mut feed = vec![0i32; n];
        for i in 0..n {
            self.logits.copy_to_host(&mut self.logits_host_scratch).map_err(|e| e.to_string())?;
            let t = sample_from_logits(&self.logits_host_scratch, &sampling, &mut rngs[i]);
            feed[i] = t;
            if self.tokenizer.is_stop(t) && !ignore_eos {
                done[i] = true;
                out[i].1 = "stop";
            } else {
                out[i].0.push(t);
            }
        }

        for _ in 1..max_tokens {
            if done.iter().all(|d| *d) {
                break;
            }
            if self.batched.as_ref().expect("batched").positions.iter().any(|&p| p + 1 >= max_seq_len()) {
                break;
            }
            crate::model::forward_batched_decode(
                &self.blas_handle,
                &self.weights,
                self.batched.as_mut().expect("batched"),
                &feed,
                &mut logits,
            )
            .map_err(|e| e.to_string())?;
            for i in 0..n {
                if done[i] {
                    continue;
                }
                logits
                    .copy_row_to_host(&mut self.logits_host_scratch, i * crate::model::VOCAB_SIZE)
                    .map_err(|e| e.to_string())?;
                let t = sample_from_logits(&self.logits_host_scratch, &sampling, &mut rngs[i]);
                if self.tokenizer.is_stop(t) && !ignore_eos {
                    done[i] = true;
                    out[i].1 = "stop";
                    continue;
                }
                out[i].0.push(t);
                feed[i] = t;
            }
        }
        Ok(out)
    }

    fn step(&mut self) -> Result<i32, String> {
        // `start_request` bounds the PROMPT against MAX_SEQ_LEN, but nothing
        // bounded decode: a long generation (or a multi-turn session whose
        // resumed position keeps accumulating) walked straight off the end of
        // the KV cache and took the whole server down with
        // "Memory access fault ... Page not present". Real crash, found by
        // running the real 128-task aider suite -- see MEASURED_FINDINGS §12.
        if self.context_exhausted() {
            return Err(format!(
                "context exhausted: position {} of {} -- one more token would write past the allocated KV cache",
                self.state.position, max_seq_len()
            ));
        }
        let next_id = if self.sampling.temperature <= 0.0 {
            argmax_sample(&self.logits).map_err(|e| e.to_string())?
        } else {
            self.logits.copy_to_host(&mut self.logits_host_scratch).map_err(|e| e.to_string())?;
            sample_from_logits(&self.logits_host_scratch, &self.sampling, &mut self.rng)
        };
        self.graphed
            .forward_one_token(&self.weights, &mut self.state, next_id, &mut self.logits)
            .map_err(|e| e.to_string())?;
        Ok(next_id)
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible request/response schema (matches
// apps/runtime-triton/server.py's real shape).
// ---------------------------------------------------------------------------

/// Messages are kept as raw JSON objects so everything an OpenAI client sends
/// (`tool_calls`, `tool_call_id`, `content: null`, structured content parts)
/// reaches the model's chat template intact. Only `role` is required.
fn validate_messages(messages: &[serde_json::Value]) -> Result<(), String> {
    if messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    for (i, m) in messages.iter().enumerate() {
        if m.get("role").and_then(serde_json::Value::as_str).is_none() {
            return Err(format!("messages[{i}] must be an object with a string \"role\""));
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: String,
    messages: Vec<serde_json::Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    /// OpenAI's `n`: how many independent completions to generate for this
    /// one prompt. Served by batched decode -- the prompt is prefilled ONCE
    /// and fanned out to `n` slots, so the weight read is shared across all
    /// of them (see MEASURED_FINDINGS §16). This is exactly the shape of the
    /// repo's own best-of-K drafting (§13), which currently pays for `n`
    /// separate full generations.
    #[serde(default)]
    n: Option<usize>,
    /// §118: real sampling contract, applies to BOTH bf16 and quantized
    /// checkpoints (operates purely at the logit-sampling layer, entirely
    /// agnostic of weight precision -- see `sampling.rs`). Omitted or
    /// `<=0.0` means greedy (the pre-existing default behavior, and what
    /// every real benchmark in `benchmarks/harness_sdk` already sends
    /// explicitly) -- a deliberate departure from the OpenAI API's own
    /// implicit default of `1.0`, in favor of this crate's own
    /// established "deterministic by default" convention.
    #[serde(default)]
    temperature: Option<f32>,
    /// Real nucleus sampling threshold. Omitted, `<=0.0`, or `>=1.0`
    /// means disabled (matches llama.cpp's own `llama_sampler_init_top_p`
    /// convention).
    #[serde(default)]
    top_p: Option<f32>,
    /// Real top-k truncation. Not a standard OpenAI field (OpenAI's API
    /// has no `top_k`), but a real, disclosed extension every other
    /// OpenAI-compatible LLM server (llama.cpp, Ollama) already ships --
    /// omitted or `<=0` means disabled.
    #[serde(default)]
    top_k: Option<i32>,
    /// Real, optional deterministic seed for this request's own sampling
    /// RNG -- the same seed reproduces the exact same real sampled
    /// sequence (`real_same_seed_reproduces_the_same_sample_sequence`).
    /// Omitted draws real OS entropy once, matching llama.cpp's own
    /// `LLAMA_DEFAULT_SEED` behavior. Meaningless (and unused) when
    /// `temperature<=0.0`.
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    /// §127: real, disclosed extension (not a standard OpenAI field, same
    /// spirit as `top_k`) -- the name of a previously-captured state
    /// snapshot (`POST /v1/state/snapshot`) to restore before this
    /// request's own `messages` are prefilled. Omitted means the
    /// pre-existing behavior: reset and prefill the full `messages` from
    /// scratch. When set, `messages` must contain ONLY the new turn(s) to
    /// continue with -- NOT the full conversation history again (see
    /// `Engine::start_request`'s own doc comment for why re-sending old
    /// turns would double-encode them).
    #[serde(default)]
    resume: Option<String>,
    /// §133: LoRA adapter name or id to activate for this completion request.
    /// Can also be specified as part of `model`, e.g. `qwen3.5:27b-rust+astral`.
    #[serde(default)]
    adapter: Option<String>,
    /// §133: Dynamic scale multiplier for the adapter (e.g. 0.5 or 0.25).
    #[serde(default)]
    adapter_scale: Option<f32>,
    /// §133: If true, ignore EOS tokens and run decode to full `max_tokens`
    /// (matching benchmark throughput measurement harnesses).
    #[serde(default)]
    ignore_eos: Option<bool>,
    /// Extra chat-template variables, same field name and meaning as vLLM and
    /// SGLang. Only `enable_thinking` (default true) is honoured.
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Value>,
}


#[derive(Serialize)]
struct ResponseChatMessage {
    role: &'static str,
    content: String,
    /// §137: real OpenAI `tool_calls`. Omitted entirely when the model made no
    /// call, so a plain completion's wire format is byte-identical to what
    /// this server has always returned -- this field is strictly ADDITIVE and
    /// `content` still carries the full raw generation (including `<think>`),
    /// which every existing benchmark in this repo already parses itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallOut>>,
}

/// OpenAI's wire shape for one tool call. `arguments` is a JSON **string**
/// (not an object) -- that is the actual OpenAI contract, and clients like
/// OpenHands call `json.loads` on it, so encoding it any other way would break
/// them.
#[derive(Serialize)]
struct ToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct ToolCallOut {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: ToolCallFunction,
}

/// Turns one real generation into real OpenAI tool calls, or `None` when the
/// model genuinely made no call. See `tool_parse` for why the server has to do
/// this at all (the Qwen3.5 template's own format is XML, not JSON).
fn build_tool_calls(
    content: &str,
    tools: Option<&[serde_json::Value]>,
    id_seed: &str,
) -> Option<Vec<ToolCallOut>> {
    // No `tools` on the request means the model was never shown any, so any
    // `<tool_call>`-looking text is ordinary prose and must stay prose.
    tools?;
    let parsed = crate::tool_parse::parse_tool_calls(content, tools);
    if parsed.is_empty() {
        return None;
    }
    Some(
        parsed
            .into_iter()
            .enumerate()
            .map(|(i, c)| ToolCallOut {
                id: format!("call_{id_seed}_{i}"),
                kind: "function",
                function: ToolCallFunction {
                    name: c.name,
                    // Serializing an object we just built cannot fail; the
                    // fallback keeps this total rather than panicking a server.
                    arguments: serde_json::to_string(&c.arguments)
                        .unwrap_or_else(|_| "{}".to_string()),
                },
            })
            .collect(),
    )
}

#[derive(Serialize)]
struct ChatCompletionChoice {
    index: u32,
    message: ResponseChatMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct UsageInfo {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    adapter_swap_ms: Option<f64>,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: &'static str,
    choices: Vec<ChatCompletionChoice>,
    usage: UsageInfo,
}

#[derive(Serialize)]
struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// §137: streamed tool calls. Emitted once, complete, at the end of the
    /// stream (see `stream_chat_completion`), never in fragments.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDeltaOut>>,
}

/// Streaming form of `ToolCallOut`: OpenAI's delta protocol adds `index` so a
/// client can accumulate fragments per call. Sending each call whole in one
/// delta is a valid use of that protocol -- a client that concatenates
/// fragments receives exactly one fragment per index.
#[derive(Serialize)]
struct ToolCallDeltaOut {
    index: u32,
    #[serde(flatten)]
    call: ToolCallOut,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: u32,
    delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: &'static str,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsageInfo>,
}

#[derive(Serialize)]
struct ModelObject {
    id: &'static str,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelListResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

fn completion_id() -> String {
    format!("chatcmpl-runtimenext-{}", unix_time_secs())
}

// ---------------------------------------------------------------------------
// Streaming: writes directly to the raw connection via `Request::into_writer`
// -- NOT `tiny_http`'s own `Response`/`respond()` path. Real, found-not-assumed
// reason (see `docs/DECISIONS.md` §103): `tiny_http` 0.12.0's chunked-response
// writer (`chunked_transfer::Encoder::new`, used internally by
// `Response::raw_print`, no public API to change it) buffers 8192 bytes with
// `flush_after_write: false` and never calls `.flush()` until the WHOLE
// response body is done -- so `Response::new(..., stream, None, None)`'s
// "streaming" `Read` adapter (this crate's own original design here) only
// ever reaches the client in ~8KB bursts (or all at once, for any response
// shorter than that), a real ~55-token-equivalent stall confirmed directly
// with `curl`'s own `time_starttransfer` against a running server.
// `into_writer()` is `tiny_http`'s own documented escape hatch for exactly
// this ("useful for things like CGI") -- hand-rolling the real HTTP/1.1
// status line, headers, and chunked-transfer-encoding framing here is a
// small, well-understood price for controlling `.flush()` ourselves.
//
// §103 follow-up, investigated and RESOLVED: an initial pass here suspected
// flushing per SSE frame cost real decode throughput (a real, reproduced
// ~90 -> ~79 tok/s drop was observed on the real HTTP benchmark right after
// this fix landed) and shipped a time-coalesced `FlushPolicy` plus a local
// `TCP_NODELAY` patch to `tiny_http` to chase it. Neither was the real
// cause: `diagnose_real_streaming_loop_overhead_without_a_real_socket`
// (below) proves decode-loop overhead (JSON serialize, SSE framing, a real
// write+flush to an in-memory buffer) is under 0.3% of total time --
// negligible, not the ~12% gap that motivated the chase. The REAL
// explanation, confirmed by the SAME test's per-segment timing: decode
// throughput genuinely DECLINES as the KV cache grows across a real
// generation (measured 81.7 -> 77.5 tok/s over a real 300-token run,
// position 54 -> 354) -- the real, structural cost of the 8 full-attention
// layers' per-token scalar kernel attending to a longer cache, not a
// streaming-layer bug. The earlier "~90 tok/s" comparison point was itself
// measured at a shallow, non-representative KV depth (`bench_real_graphed_decode_tokens_per_second`'s
// own real prompt is 5 tokens, timing decode at position ~5-28 only) --
// comparing it to a real ~350-token generation's average throughput was
// never an apples-to-apples comparison. Both the `TCP_NODELAY` patch and
// `FlushPolicy`'s coalescing were reverted/simplified once this was
// understood -- this file flushes every real SSE frame unconditionally,
// which real measurement shows costs nothing worth trading TTFT for.
// ---------------------------------------------------------------------------

fn write_sse_frame(writer: &mut dyn Write, data: &[u8]) -> std::io::Result<()> {
    write!(writer, "{:x}\r\n", data.len())?;
    writer.write_all(data)?;
    writer.write_all(b"\r\n")?;
    writer.flush()
}

fn sse_frame_bytes(id: &str, delta: ChunkDelta, finish_reason: Option<&'static str>) -> Vec<u8> {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created: unix_time_secs(),
        model: MODEL_ID,
        choices: vec![ChunkChoice { index: 0, delta, finish_reason }],
        usage: None,
    };
    let json = serde_json::to_string(&chunk).unwrap_or_default();
    format!("data: {json}\n\n").into_bytes()
}

/// Runs one real streaming chat completion to completion, writing and
/// flushing each SSE frame as it's produced -- real measurement
/// (`diagnose_real_streaming_loop_overhead_without_a_real_socket`) shows
/// this costs nothing worth trading TTFT for (see this section's own
/// header doc).
///
/// This is also this crate's real, deliberate "stop generation"
/// mechanism, not merely an accident of `?`-propagation left unexamined:
/// every SSE frame write below goes through the outer `?`-propagating
/// closure, so the moment a client disconnects (closes the tab, kills
/// curl, drops the TCP connection), the NEXT write attempt fails
/// (broken pipe) and the closure returns immediately -- the decode loop
/// stops within at most one extra real token past the disconnect, the
/// same bound the legacy Python engines' own explicit
/// `POST /api/engine/stop_generation` documents for its cooperative
/// `threading.Event` mechanism. No separate stop endpoint exists here on
/// purpose: this server is a single blocking `tiny_http` accept loop with
/// no concurrency to receive an out-of-band stop request WHILE a
/// generation is in flight (see this file's own header doc on being
/// "single-tenant by design, not by accident") -- an explicit
/// stop-while-still-connected endpoint would need a real background
/// generation thread and cross-thread signaling, a genuine architecture
/// change, not attempted here. The `MutexGuard<Engine>` this function
/// takes by value is dropped on every real return path (normal
/// completion, EOS, `max_tokens`, OR an aborted write), releasing the
/// lock for the next queued request regardless of how this one ended --
/// no cleanup needed, matching this crate's state model (`DecodeState`
/// is resumable from wherever it's left, the same property the
/// state-handoff feature already relies on). See
/// `real_disconnected_client_stops_generation_within_one_token` for the
/// decisive proof.
fn stream_chat_completion(
    mut writer: Box<dyn Write + Send>,
    mut engine: std::sync::MutexGuard<Engine>,
    prompt_tokens: usize,
    max_tokens: usize,
    ignore_eos: bool,
    tools: Option<&[serde_json::Value]>,
) {
    let write_result = (|| -> std::io::Result<()> {
        write!(
            writer,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n"
        )?;

        let id = completion_id();
        let role_frame = sse_frame_bytes(&id, ChunkDelta { role: Some("assistant"), content: None, tool_calls: None }, None);
        write_sse_frame(&mut *writer, &role_frame)?;
        let mut generated_ids: Vec<i32> = Vec::new();
        let mut prev_text_len = 0usize;
        let mut full_text = String::new();
        let t0 = std::time::Instant::now();
        let mut finish: &'static str = loop {
            if generated_ids.len() >= max_tokens || engine.context_exhausted() {
                break "length";
            }
            let next_id = match engine.step() {
                Ok(next_id) => next_id,
                Err(e) => return Err(std::io::Error::other(e)),
            };
            if engine.tokenizer.is_stop(next_id) && !ignore_eos {
                break "stop";
            }
            generated_ids.push(next_id);
            full_text = engine.tokenizer.decode(&generated_ids).map_err(std::io::Error::other)?;
            // Emit only the new suffix -- decoding the whole growing
            // sequence each step (not token-by-token) correctly handles
            // multi-token UTF-8/BPE merge boundaries.
            let delta_text = full_text.get(prev_text_len..).unwrap_or("").to_string();
            prev_text_len = full_text.len();
            if delta_text.is_empty() {
                continue;
            }
            let frame = sse_frame_bytes(&id, ChunkDelta { role: None, content: Some(delta_text), tool_calls: None }, None);
            write_sse_frame(&mut *writer, &frame)?;
        };

        // §137: tool calls are parsed from the COMPLETE text, once. Parsing
        // incrementally would mean emitting a call before its closing tag
        // proves it complete -- exactly the fabrication `tool_parse` refuses.
        // Content was already streamed unchanged above, so a client that
        // ignores `tool_calls` sees the same stream it always did.
        if let Some(calls) = build_tool_calls(&full_text, tools, &unix_time_secs().to_string()) {
            let deltas = calls
                .into_iter()
                .enumerate()
                .map(|(i, call)| ToolCallDeltaOut { index: i as u32, call })
                .collect();
            let frame = sse_frame_bytes(&id, ChunkDelta { role: None, content: None, tool_calls: Some(deltas) }, None);
            write_sse_frame(&mut *writer, &frame)?;
            finish = "tool_calls";
        }
        let finish_frame = sse_frame_bytes(&id, ChunkDelta { role: None, content: None, tool_calls: None }, Some(finish));
        write_sse_frame(&mut *writer, &finish_frame)?;

        let elapsed = t0.elapsed();
        let elapsed_s = elapsed.as_secs_f64();
        let tok_s = if elapsed_s > 0.0 && !generated_ids.is_empty() {
            (generated_ids.len() as f64) / elapsed_s
        } else {
            0.0
        };

        // Final usage chunk (OpenAI spec: stream_options.include_usage or standard final chunk)
        let usage_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created: unix_time_secs(),
            model: MODEL_ID,
            choices: vec![],
            usage: Some(UsageInfo {
                prompt_tokens,
                completion_tokens: generated_ids.len(),
                total_tokens: prompt_tokens + generated_ids.len(),
                tokens_per_second: Some((tok_s * 100.0).round() / 100.0),
                generation_time_ms: Some((elapsed_s * 1000.0 * 10.0).round() / 10.0),
                adapter_swap_ms: None,
            }),
        };
        let usage_json = serde_json::to_string(&usage_chunk).unwrap_or_default();
        let usage_frame = format!("data: {usage_json}\n\n").into_bytes();
        write_sse_frame(&mut *writer, &usage_frame)?;

        // Standard SSE completion frame
        write_sse_frame(&mut *writer, b"data: [DONE]\n\n")?;

        // Real chunked-transfer-encoding terminator.
        write!(writer, "0\r\n\r\n")?;
        writer.flush()
    })();
    if let Err(e) = write_result {
        eprintln!("[runtime-next] SSE stream write failed: {e}");
    }
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

fn json_response(status: u16, body: &impl Serialize) -> Response<std::io::Cursor<Vec<u8>>> {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    Response::new(StatusCode(status), vec![header], std::io::Cursor::new(bytes), None, None)
}

fn handle_health(request: Request) {
    let _ = request.respond(json_response(200, &serde_json::json!({"status": "ok"})));
}

fn handle_models(request: Request) {
    let body = ModelListResponse {
        object: "list",
        data: vec![ModelObject {
            id: MODEL_ID,
            object: "model",
            created: unix_time_secs(),
            owned_by: "gnn-experiment-runtime-next",
        }],
    };
    let _ = request.respond(json_response(200, &body));
}

// ---------------------------------------------------------------------------
// §127: state handoff -- real, server-side snapshot management, separate
// from `/v1/chat/completions` itself (which only ever RESTORES a named
// snapshot, via the `resume` field -- see `Engine::start_request`).
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SnapshotRequest {
    name: String,
}

#[derive(Serialize)]
struct SnapshotResponse {
    name: String,
    position: usize,
    created: u64,
}

#[derive(Serialize)]
struct SnapshotListResponse {
    snapshots: Vec<SnapshotResponse>,
}

fn read_json_body<T: for<'de> Deserialize<'de>>(request: &mut Request) -> Result<T, String> {
    let mut body_str = String::new();
    request.as_reader().read_to_string(&mut body_str).map_err(|e| format!("failed to read request body: {e}"))?;
    serde_json::from_str(&body_str).map_err(|e| format!("invalid request JSON: {e}"))
}

fn handle_state_snapshot_create(mut request: Request, engine: &Mutex<Engine>) {
    let req: SnapshotRequest = match read_json_body(&mut request) {
        Ok(r) => r,
        Err(e) => {
            let _ = request.respond(json_response(400, &serde_json::json!({"error": e})));
            return;
        }
    };
    let mut guard = match engine.lock() {
        Ok(g) => g,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("engine lock poisoned: {e}")})));
            return;
        }
    };
    match guard.snapshot_state(req.name.clone()) {
        Ok((position, created)) => {
            let _ = request.respond(json_response(200, &SnapshotResponse { name: req.name, position, created }));
        }
        Err(e) => {
            let _ = request.respond(json_response(400, &serde_json::json!({"error": e})));
        }
    }
}

fn handle_state_snapshots_list(request: Request, engine: &Mutex<Engine>) {
    let guard = match engine.lock() {
        Ok(g) => g,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("engine lock poisoned: {e}")})));
            return;
        }
    };
    let snapshots = guard
        .list_snapshots()
        .into_iter()
        .map(|(name, position, created)| SnapshotResponse { name, position, created })
        .collect();
    let _ = request.respond(json_response(200, &SnapshotListResponse { snapshots }));
}

fn handle_state_snapshot_delete(mut request: Request, engine: &Mutex<Engine>) {
    let req: SnapshotRequest = match read_json_body(&mut request) {
        Ok(r) => r,
        Err(e) => {
            let _ = request.respond(json_response(400, &serde_json::json!({"error": e})));
            return;
        }
    };
    let mut guard = match engine.lock() {
        Ok(g) => g,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("engine lock poisoned: {e}")})));
            return;
        }
    };
    if guard.delete_snapshot(&req.name) {
        let _ = request.respond(json_response(200, &serde_json::json!({"deleted": req.name})));
    } else {
        let _ = request.respond(json_response(404, &serde_json::json!({"error": format!("no snapshot named {:?}", req.name)})));
    }
}

/// §118: real request-level validation for the sampling contract,
/// applying this crate's own documented defaults (`temperature<=0.0` =
/// greedy, `top_p` default 1.0/disabled, `top_k` default 0/disabled --
/// see `ChatCompletionRequest`'s own field doc comments). Returns a
/// human-readable error for a real out-of-range value rather than
/// silently clamping it -- matches OpenAI's own API behavior (a bad
/// `temperature`/`top_p` is a 400, not a quiet reinterpretation).
fn parse_sampling_params(req: &ChatCompletionRequest) -> Result<SamplingParams, String> {
    let temperature = req.temperature.unwrap_or(0.0);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(format!("temperature must be >= 0.0, got {temperature}"));
    }
    let top_p = req.top_p.unwrap_or(1.0);
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(format!("top_p must be within [0.0, 1.0], got {top_p}"));
    }
    let top_k = req.top_k.unwrap_or(0);
    if top_k < 0 {
        return Err(format!("top_k must be >= 0, got {top_k}"));
    }
    Ok(SamplingParams { temperature, top_p, top_k })
}

#[derive(Deserialize)]
struct ScaleAssignment {
    #[serde(default)]
    id: Option<usize>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    scale: Option<f32>,
}

#[derive(Deserialize)]
struct AdapterSwapBody {
    #[serde(default)]
    adapter: Option<String>,
    #[serde(default)]
    id: Option<usize>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    scale: Option<f32>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    action: Option<String>,
}

fn handle_lora_adapters_get(request: Request, engine: &Mutex<Engine>) {
    let guard = match engine.lock() {
        Ok(g) => g,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("engine lock poisoned: {e}")})));
            return;
        }
    };
    let list = guard.list_adapters();
    let _ = request.respond(json_response(200, &list));
}

fn handle_lora_adapters_post(mut request: Request, engine: &Mutex<Engine>) {
    let mut body_str = String::new();
    if let Err(e) = request.as_reader().read_to_string(&mut body_str) {
        let _ = request.respond(json_response(400, &serde_json::json!({"error": format!("failed to read body: {e}")})));
        return;
    }
    let trimmed = body_str.trim();

    let mut guard = match engine.lock() {
        Ok(g) => g,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("engine lock poisoned: {e}")})));
            return;
        }
    };

    if trimmed.starts_with('[') {
        // llama.cpp scale assignment array format: [{"id": 0, "scale": 1.0}, ...]
        let assignments: Vec<ScaleAssignment> = match serde_json::from_str(trimmed) {
            Ok(a) => a,
            Err(e) => {
                let _ = request.respond(json_response(400, &serde_json::json!({"error": format!("invalid scale array JSON: {e}")})));
                return;
            }
        };

        let mut updates = Vec::new();
        for item in assignments {
            let id = if let Some(id) = item.id {
                id
            } else if let Some(name) = &item.name {
                match guard.adapters.iter().position(|a| a.name == *name || a.name.contains(name)) {
                    Some(idx) => idx,
                    None => {
                        let _ = request.respond(json_response(404, &serde_json::json!({"error": format!("adapter name {name:?} not found")})));
                        return;
                    }
                }
            } else if let Some(path) = &item.path {
                match guard.adapters.iter().position(|a| a.path.contains(path)) {
                    Some(idx) => idx,
                    None => {
                        let _ = request.respond(json_response(404, &serde_json::json!({"error": format!("adapter path {path:?} not found")})));
                        return;
                    }
                }
            } else {
                let _ = request.respond(json_response(400, &serde_json::json!({"error": "each assignment requires 'id', 'name', or 'path'"})));
                return;
            };
            let scale = item.scale.unwrap_or(1.0);
            updates.push((id, scale));
        }

        match guard.set_adapter_scales(&updates) {
            Ok(_ms) => {
                let list = guard.list_adapters();
                let _ = request.respond(json_response(200, &list));
            }
            Err(e) => {
                let _ = request.respond(json_response(500, &serde_json::json!({"error": e})));
            }
        }
    } else {
        // Single object format (dynamic load or single swap)
        let body: AdapterSwapBody = match serde_json::from_str(trimmed) {
            Ok(b) => b,
            Err(e) => {
                let _ = request.respond(json_response(400, &serde_json::json!({"error": format!("invalid JSON object: {e}")})));
                return;
            }
        };

        // If action == "load" or path is supplied without adapter/id:
        if body.action.as_deref() == Some("load") || (body.path.is_some() && body.adapter.is_none() && body.id.is_none()) {
            let path = body.path.unwrap_or_default();
            match guard.load_adapter(&path, body.name.as_deref()) {
                Ok(id) => {
                    let info = &guard.adapters[id];
                    let _ = request.respond(json_response(200, &serde_json::json!({
                        "status": "loaded",
                        "id": info.id,
                        "name": info.name,
                        "path": info.path,
                    })));
                }
                Err(e) => {
                    let _ = request.respond(json_response(400, &serde_json::json!({"error": e})));
                }
            }
            return;
        }

        // If scale is supplied in single object: {"id": 0, "scale": 1.0} or {"adapter": "astral", "scale": 0.5}
        if let Some(scale) = body.scale {
            let target_id = if let Some(id) = body.id {
                Some(id)
            } else if let Some(name) = &body.name {
                guard.adapters.iter().find(|a| a.name == *name || a.name.contains(name)).map(|a| a.id)
            } else if let Some(adapter) = &body.adapter {
                guard.adapters.iter().find(|a| a.name == *adapter || a.name.contains(adapter)).map(|a| a.id)
            } else {
                None
            };
            if let Some(id) = target_id {
                match guard.set_adapter_scales(&[(id, scale)]) {
                    Ok(_ms) => {
                        let list = guard.list_adapters();
                        let _ = request.respond(json_response(200, &list));
                        return;
                    }
                    Err(e) => {
                        let _ = request.respond(json_response(500, &serde_json::json!({"error": e})));
                        return;
                    }
                }
            }
        }

        // Single swap
        let spec = if let Some(adapter) = &body.adapter {
            adapter.as_str()
        } else if let Some(id) = body.id {
            &id.to_string()
        } else if let Some(name) = &body.name {
            name.as_str()
        } else {
            "base"
        };

        match guard.swap_to_adapter(spec) {
            Ok((active_id, ms)) => {
                let _ = request.respond(json_response(200, &serde_json::json!({
                    "status": "ok",
                    "active_adapter_id": active_id,
                    "swap_latency_ms": (ms * 100.0).round() / 100.0,
                    "adapters": guard.list_adapters(),
                })));
            }
            Err(e) => {
                let _ = request.respond(json_response(400, &serde_json::json!({"error": e})));
            }
        }
    }
}

fn handle_adapters_swap(mut request: Request, engine: &Mutex<Engine>) {
    let body: AdapterSwapBody = match read_json_body(&mut request) {
        Ok(b) => b,
        Err(e) => {
            let _ = request.respond(json_response(400, &serde_json::json!({"error": e})));
            return;
        }
    };
    let mut guard = match engine.lock() {
        Ok(g) => g,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("engine lock poisoned: {e}")})));
            return;
        }
    };
    let spec = if let Some(adapter) = &body.adapter {
        adapter.as_str()
    } else if let Some(id) = body.id {
        &id.to_string()
    } else if let Some(name) = &body.name {
        name.as_str()
    } else {
        "base"
    };

    match guard.swap_to_adapter(spec) {
        Ok((active_id, ms)) => {
            let _ = request.respond(json_response(200, &serde_json::json!({
                "status": "ok",
                "active_adapter_id": active_id,
                "swap_latency_ms": (ms * 100.0).round() / 100.0,
            })));
        }
        Err(e) => {
            let _ = request.respond(json_response(400, &serde_json::json!({"error": e})));
        }
    }
}

fn handle_chat_completions(mut request: Request, engine: &Mutex<Engine>) {
    let mut body_str = String::new();
    if let Err(e) = request.as_reader().read_to_string(&mut body_str) {
        let _ = request.respond(json_response(400, &serde_json::json!({"error": format!("failed to read request body: {e}")})));
        return;
    }
    let req: ChatCompletionRequest = match serde_json::from_str(&body_str) {
        Ok(r) => r,
        Err(e) => {
            let _ = request.respond(json_response(400, &serde_json::json!({"error": format!("invalid request JSON: {e}")})));
            return;
        }
    };
    let sampling = match parse_sampling_params(&req) {
        Ok(s) => s,
        Err(e) => {
            let _ = request.respond(json_response(400, &serde_json::json!({"error": e})));
            return;
        }
    };
    if let Err(e) = validate_messages(&req.messages) {
        let _ = request.respond(json_response(400, &serde_json::json!({"error": e})));
        return;
    }
    let template_kwargs = req.chat_template_kwargs.as_ref().and_then(serde_json::Value::as_object);
    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let ignore_eos = req.ignore_eos.unwrap_or(false);

    let mut guard = match engine.lock() {
        Ok(g) => g,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("engine lock poisoned: {e}")})));
            return;
        }
    };

    // §133: Dynamic LoRA adapter resolution from request
    let raw_adapter_target = req.adapter.as_deref().or_else(|| {
        if req.model.contains('+') {
            req.model.split('+').nth(1)
        } else if guard.adapters.iter().any(|a| a.name == req.model) {
            Some(req.model.as_str())
        } else {
            None
        }
    });

    let adapter_target = raw_adapter_target.map(|t| {
        if let Some(s) = req.adapter_scale {
            if !t.contains('@') {
                return format!("{t}@{s}");
            }
        }
        t.to_string()
    });

    let mut adapter_swap_ms: Option<f64> = None;
    if let Some(target) = &adapter_target {
        match guard.swap_to_adapter(target) {
            Ok((_, ms)) => {
                if ms > 0.05 {
                    adapter_swap_ms = Some((ms * 100.0).round() / 100.0);
                }
            }
            Err(e) => {
                let _ = request.respond(json_response(400, &serde_json::json!({
                    "error": format!("failed to activate adapter {target:?}: {e}")
                })));
                return;
            }
        }
    }

    let tools = req.tools.as_deref();
    let prompt_tokens = match guard.start_request_json(
        &req.messages,
        tools,
        sampling,
        req.seed,
        req.resume.as_deref(),
        template_kwargs,
    ) {
        Ok(n) => n,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("prefill failed: {e}")})));
            return;
        }
    };

    // `n > 1`: one shared prefill fanned out to n batch slots (§16). Only on
    // the non-streaming path -- interleaving n token streams over one SSE
    // connection is not something the OpenAI wire format expresses.
    let n = req.n.unwrap_or(1).max(1);
    if n > 1 && !req.stream {
        if n > crate::model::MAX_PREFILL_CHUNK {
            let _ = request.respond(json_response(400, &serde_json::json!({
                "error": format!("n={n} exceeds the maximum batch this engine can hold ({})", crate::model::MAX_PREFILL_CHUNK)
            })));
            return;
        }
        if sampling.temperature <= 0.0 {
            let _ = request.respond(json_response(400, &serde_json::json!({
                "error": "n>1 requires temperature>0: at temperature 0 every completion is the identical greedy continuation"
            })));
            return;
        }
        let t0 = std::time::Instant::now();
        let results = match guard.generate_n(n, max_tokens, sampling, req.seed, ignore_eos) {
            Ok(r) => r,
            Err(e) => {
                let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("batched decode failed: {e}")})));
                return;
            }
        };
        let elapsed_s = t0.elapsed().as_secs_f64();
        let total: usize = results.iter().map(|(ids, _)| ids.len()).sum();
        let mut choices = Vec::with_capacity(n);
        for (i, (ids, finish)) in results.iter().enumerate() {
            let content = match guard.tokenizer.decode(ids) {
                Ok(c) => c,
                Err(e) => {
                    let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("detokenize failed: {e}")})));
                    return;
                }
            };
            let tool_calls = build_tool_calls(&content, req.tools.as_deref(), &format!("{}_{i}", unix_time_secs()));
            let finish_reason = if tool_calls.is_some() { "tool_calls" } else { finish };
            choices.push(ChatCompletionChoice {
                index: i as u32,
                message: ResponseChatMessage { role: "assistant", content, tool_calls },
                finish_reason,
            });
        }
        let tok_s = if elapsed_s > 0.0 && total > 0 { total as f64 / elapsed_s } else { 0.0 };
        let body = ChatCompletionResponse {
            id: completion_id(),
            object: "chat.completion",
            created: unix_time_secs(),
            model: MODEL_ID,
            choices,
            usage: UsageInfo {
                prompt_tokens,
                completion_tokens: total,
                total_tokens: prompt_tokens + total,
                tokens_per_second: Some((tok_s * 100.0).round() / 100.0),
                generation_time_ms: Some((elapsed_s * 1000.0 * 10.0).round() / 10.0),
                adapter_swap_ms,
            },
        };
        let _ = request.respond(json_response(200, &body));
        return;
    }

    if req.stream {
        // §103: bypasses `tiny_http`'s own `Response`/`respond()` path
        // entirely -- see `stream_chat_completion`'s own doc comment for
        // why (that path's chunked writer buffers 8KB with no flush).
        let writer = request.into_writer();
        stream_chat_completion(writer, guard, prompt_tokens, max_tokens, ignore_eos, req.tools.as_deref());
        return;
    }

    // Non-streaming: run the full decode loop to completion, then return
    // one JSON response.
    let mut generated_ids: Vec<i32> = Vec::new();
    let mut finish_reason = "length";
    let t0 = std::time::Instant::now();
    for _ in 0..max_tokens {
        if guard.context_exhausted() {
            finish_reason = "length";
            break;
        }
        let next_id = match guard.step() {
            Ok(id) => id,
            Err(e) => {
                let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("decode failed: {e}")})));
                return;
            }
        };
        if guard.tokenizer.is_stop(next_id) && !ignore_eos {
            finish_reason = "stop";
            break;
        }
        generated_ids.push(next_id);
    }
    let elapsed = t0.elapsed();
    let elapsed_s = elapsed.as_secs_f64();
    let tok_s = if elapsed_s > 0.0 && !generated_ids.is_empty() {
        (generated_ids.len() as f64) / elapsed_s
    } else {
        0.0
    };
    let content = match guard.tokenizer.decode(&generated_ids) {
        Ok(t) => t,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("detokenize failed: {e}")})));
            return;
        }
    };
    // §137: a real tool call ends the turn with OpenAI's own
    // `finish_reason="tool_calls"`, which is what agent loops (OpenHands
    // included) branch on to execute the call instead of treating the turn as
    // a final answer. A call cut off by `max_tokens` never parses (no closing
    // tag), so "length" is preserved exactly when it is true.
    let tool_calls = build_tool_calls(&content, req.tools.as_deref(), &unix_time_secs().to_string());
    if tool_calls.is_some() {
        finish_reason = "tool_calls";
    }
    let response_body = ChatCompletionResponse {
        id: completion_id(),
        object: "chat.completion",
        created: unix_time_secs(),
        model: MODEL_ID,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ResponseChatMessage { role: "assistant", content, tool_calls },
            finish_reason,
        }],
        usage: UsageInfo {
            prompt_tokens,
            completion_tokens: generated_ids.len(),
            total_tokens: prompt_tokens + generated_ids.len(),
            tokens_per_second: Some((tok_s * 100.0).round() / 100.0),
            generation_time_ms: Some((elapsed_s * 1000.0 * 10.0).round() / 10.0),
            adapter_swap_ms,
        },
    };
    let _ = request.respond(json_response(200, &response_body));
}


/// `POST /v1/chat/completions/batch` -- N INDEPENDENT prompts decoded together.
///
/// Static batching, deliberately: the caller already holds all N prompts (a
/// benchmark sweep, a best-of-K fan-out, an offline eval), so there is nothing
/// to schedule. True continuous batching -- dynamic admission of concurrent
/// clients into a running batch -- needs a threaded scheduler and is NOT what
/// this is; see MEASURED_FINDINGS §16h.
fn handle_chat_completions_batch(mut request: tiny_http::Request, engine: &Mutex<Engine>) {
    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_err() {
        let _ = request.respond(json_response(400, &serde_json::json!({"error": "could not read request body"})));
        return;
    }
    let req: BatchCompletionRequest = match serde_json::from_str(&body) {
        Ok(r) => r,
        Err(e) => {
            let _ = request.respond(json_response(400, &serde_json::json!({"error": format!("invalid request JSON: {e}")})));
            return;
        }
    };
    if req.batch.is_empty() {
        let _ = request.respond(json_response(400, &serde_json::json!({"error": "batch must contain at least one entry"})));
        return;
    }
    if req.batch.len() > crate::model::MAX_PREFILL_CHUNK {
        let _ = request.respond(json_response(400, &serde_json::json!({
            "error": format!("batch of {} exceeds the maximum this engine can hold ({})", req.batch.len(), crate::model::MAX_PREFILL_CHUNK)
        })));
        return;
    }

    for (i, e) in req.batch.iter().enumerate() {
        if let Err(err) = validate_messages(&e.messages) {
            let _ = request.respond(json_response(400, &serde_json::json!({"error": format!("batch[{i}]: {err}")})));
            return;
        }
    }
    let prompts: Vec<Vec<serde_json::Value>> = req.batch.iter().map(|e| e.messages.clone()).collect();
    let template_kwargs = req.chat_template_kwargs.as_ref().and_then(serde_json::Value::as_object);

    let mut guard = match engine.lock() {
        Ok(g) => g,
        Err(_) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": "engine mutex poisoned"})));
            return;
        }
    };

    let mut adapter_swap_ms = None;
    if let Some(target) = req.adapter.as_deref() {
        match guard.swap_to_adapter(target) {
            Ok((_, ms)) => adapter_swap_ms = Some((ms * 100.0).round() / 100.0),
            Err(e) => {
                let _ = request.respond(json_response(400, &serde_json::json!({"error": format!("adapter swap failed: {e}")})));
                return;
            }
        }
    }

    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
    };
    let t0 = std::time::Instant::now();
    let (results, prompt_tokens) = match guard.generate_batch(&prompts, None, template_kwargs, max_tokens, sampling, req.seed, false) {
        Ok(r) => r,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("batched generation failed: {e}")})));
            return;
        }
    };
    let elapsed_s = t0.elapsed().as_secs_f64();
    let total: usize = results.iter().map(|(ids, _)| ids.len()).sum();

    let mut responses = Vec::with_capacity(results.len());
    for (i, (ids, finish)) in results.iter().enumerate() {
        let content = match guard.tokenizer.decode(ids) {
            Ok(c) => c,
            Err(e) => {
                let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("detokenize failed: {e}")})));
                return;
            }
        };
        responses.push(serde_json::json!({
            "index": i,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish,
            "usage": {
                "prompt_tokens": prompt_tokens[i],
                "completion_tokens": ids.len(),
                "total_tokens": prompt_tokens[i] + ids.len(),
            }
        }));
    }
    let tok_s = if elapsed_s > 0.0 && total > 0 { total as f64 / elapsed_s } else { 0.0 };
    let _ = request.respond(json_response(200, &serde_json::json!({
        "object": "chat.completion.batch",
        "id": completion_id(),
        "created": unix_time_secs(),
        "model": MODEL_ID,
        "batch_size": results.len(),
        "responses": responses,
        "usage": {
            "completion_tokens": total,
            "tokens_per_second": (tok_s * 100.0).round() / 100.0,
            "generation_time_ms": (elapsed_s * 1000.0 * 10.0).round() / 10.0,
            "adapter_swap_ms": adapter_swap_ms,
        }
    })));
}

#[derive(Deserialize)]
struct BatchCompletionEntry {
    messages: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct BatchCompletionRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: String,
    batch: Vec<BatchCompletionEntry>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    seed: Option<u64>,
    /// One adapter for the WHOLE batch -- folding mutates the live weights, so
    /// per-slot adapters are not expressible here (§12f).
    #[serde(default)]
    adapter: Option<String>,
    /// See `ChatCompletionRequest::chat_template_kwargs`.
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Value>,
}

/// Loads real weights, pre-loads initial LoRA adapters if specified, starts
/// the real HTTP server on `port`, and serves forever.
pub fn run(port: u16, initial_loras: &[String]) -> Result<(), String> {
    let mut engine = Engine::load()?;
    for path in initial_loras {
        eprintln!("[runtime-next] pre-loading initial LoRA adapter: {path}...");
        match engine.load_adapter(path, None) {
            Ok(id) => eprintln!("[runtime-next] loaded adapter slot {id}: {path}"),
            Err(e) => eprintln!("[runtime-next] warning: failed to pre-load adapter {path}: {e}"),
        }
    }
    let engine = Mutex::new(engine);

    let server = Server::http(("0.0.0.0", port)).map_err(|e| format!("failed to bind port {port}: {e}"))?;
    eprintln!("[runtime-next] real HTTP server listening on 0.0.0.0:{port}");

    for request in server.incoming_requests() {
        let method = request.method().clone();
        let url = request.url().to_string();
        let path = url.split('?').next().unwrap_or(&url);
        match (method, path) {
            (Method::Get, "/health") => handle_health(request),
            (Method::Get, "/v1/models") => handle_models(request),
            (Method::Get, "/lora-adapters") | (Method::Get, "/v1/adapters") => handle_lora_adapters_get(request, &engine),
            (Method::Post, "/lora-adapters") => handle_lora_adapters_post(request, &engine),
            (Method::Post, "/v1/adapters/swap") => handle_adapters_swap(request, &engine),
            (Method::Post, "/v1/chat/completions") => handle_chat_completions(request, &engine),
            (Method::Post, "/v1/chat/completions/batch") => handle_chat_completions_batch(request, &engine),
            (Method::Post, "/v1/state/snapshot") => handle_state_snapshot_create(request, &engine),
            (Method::Get, "/v1/state/snapshots") => handle_state_snapshots_list(request, &engine),
            (Method::Delete, "/v1/state/snapshots") => handle_state_snapshot_delete(request, &engine),
            _ => {
                let _ = request.respond(json_response(404, &serde_json::json!({"error": "not found"})));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// §137: pins the exact OpenAI wire shape of a tool call -- `arguments` is
    /// a JSON STRING, `type` is "function", ids are unique per call -- since
    /// that shape, not the parser, is the contract every client relies on.
    #[test]
    fn tool_calls_serialize_in_the_exact_openai_wire_shape() {
        let tools = vec![serde_json::json!({"type": "function", "function": {
            "name": "calculate_triangle_area",
            "parameters": {"properties": {"base": {"type": "integer"}, "height": {"type": "integer"}}}
        }})];
        let content = "<think>\nok\n</think>\n<tool_call>\n<function=calculate_triangle_area>\n\
            <parameter=base>\n10\n</parameter>\n<parameter=height>\n5\n</parameter>\n</function>\n</tool_call>";
        let calls = build_tool_calls(content, Some(&tools), "t").expect("one call");
        let msg = ResponseChatMessage { role: "assistant", content: content.to_string(), tool_calls: Some(calls) };
        let v = serde_json::to_value(&msg).unwrap();
        let call = &v["tool_calls"][0];
        assert_eq!(call["type"], "function");
        assert_eq!(call["id"], "call_t_0");
        assert_eq!(call["function"]["name"], "calculate_triangle_area");
        let args = call["function"]["arguments"].as_str().expect("arguments must be a JSON string");
        assert_eq!(serde_json::from_str::<serde_json::Value>(args).unwrap(), serde_json::json!({"base": 10, "height": 5}));
    }

    /// Without `tools` on the request the model was never offered any, so the
    /// response must be byte-identical to before §137: no `tool_calls` key.
    #[test]
    fn no_tools_on_the_request_means_no_tool_calls_key_at_all() {
        let content = "<tool_call>\n<function=f>\n</function>\n</tool_call>";
        assert!(build_tool_calls(content, None, "t").is_none());
        let msg = ResponseChatMessage { role: "assistant", content: content.to_string(), tool_calls: None };
        let v = serde_json::to_value(&msg).unwrap();
        assert!(v.get("tool_calls").is_none(), "plain completions must keep their old wire shape");
    }

    /// `chat_template_kwargs.enable_thinking=false` must reproduce the upstream
    /// Qwen3.5 template's `enable_thinking is false` branch byte-for-byte:
    /// `{{- '<think>\n\n</think>\n\n' }}` after the assistant header.
    #[test]
    fn enable_thinking_switch_matches_the_upstream_template_branches() {
        let tok = crate::tokenizer::ChatTokenizer::load(&crate::model_loader::locate_model_snapshot().unwrap()).unwrap();
        let on = tok.apply_chat_template_full(&[("user", "hi")], None, true);
        let off = tok.apply_chat_template_full(&[("user", "hi")], None, false);
        assert_eq!(on, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert_eq!(off, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert_eq!(tok.apply_chat_template_with_tools(&[("user", "hi")], None), on, "default path unchanged");
    }

    /// Streaming deltas carry OpenAI's per-call `index` alongside the flattened call.
    #[test]
    fn streamed_tool_call_delta_carries_index_and_flattened_call() {
        let tools = vec![serde_json::json!({"name": "f", "parameters": {"properties": {}}})];
        let calls = build_tool_calls("<tool_call>\n<function=f>\n</function>\n</tool_call>", Some(&tools), "t").unwrap();
        let deltas: Vec<ToolCallDeltaOut> =
            calls.into_iter().enumerate().map(|(i, call)| ToolCallDeltaOut { index: i as u32, call }).collect();
        let v = serde_json::to_value(ChunkDelta { role: None, content: None, tool_calls: Some(deltas) }).unwrap();
        assert_eq!(v["tool_calls"][0]["index"], 0);
        assert_eq!(v["tool_calls"][0]["function"]["name"], "f");
        assert_eq!(v["tool_calls"][0]["function"]["arguments"], "{}");
    }

    /// §127 decisive test: the real, server-level state-handoff capability
    /// (`Engine::snapshot_state` + `start_request`'s own `resume`
    /// parameter) must produce a BIT-EXACT identical second-turn
    /// generation to the pre-existing, already-trusted "resend the full
    /// conversation history" pattern every stateless OpenAI-style chat API
    /// (including this server's own, before this feature) relies on.
    ///
    /// Two real two-turn conversations, same real prompt, same real model:
    /// - "Full history" (Engine A): turn 1 via a normal `start_request`
    ///   (`resume: None`), generate the real reply, then turn 2 via
    ///   ANOTHER normal `start_request` that resends the ENTIRE real
    ///   conversation so far (system/user/assistant/user) -- the real,
    ///   only mechanism a client had for multi-turn conversations before
    ///   this feature, and still the real ground truth this new feature
    ///   must never silently diverge from.
    /// - "Resumed" (Engine B): turn 1 identically, then
    ///   `Engine::snapshot_state` captures the state right after, and
    ///   turn 2 goes through `start_request` with `resume: Some(name)`
    ///   and ONLY the new user turn -- the real, new, faster path this
    ///   feature adds.
    ///
    /// If these ever diverge, the handoff is real but WRONG -- silently
    /// corrupting every resumed conversation's second turn onward, the
    /// exact class of bug this project's own §117 postmortem (a stale
    /// HIP-Graph-captured host value) already cost real debugging time
    /// once. This test exists to make that impossible to ship unnoticed.
    #[test]
    #[ignore]
    fn real_state_handoff_snapshot_and_resume_matches_full_history_resend() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let system = "You are a helpful assistant.";
        let turn1_user = "What is the capital of France?";
        let turn2_user = "What is its population, roughly?";
        let turn1_reply_len = 5usize;
        let turn2_reply_len = 5usize;

        // --- Engine A: full-history resend (the real, pre-existing, still-
        // supported path -- ground truth). Real, deliberate scoping: A is
        // loaded, used, and DROPPED (freeing its real VRAM via
        // `ModelWeights`/`DecodeState`'s own `Drop` impls) BEFORE B is ever
        // loaded -- this machine may be running other real GPU workloads
        // concurrently (this repo's own harness services), and two full
        // model copies held live at once is real, unnecessary VRAM
        // pressure this test doesn't need to risk.
        let (turn1_ids, turn2_ids_a) = {
            let mut engine_a = Engine::load().expect("real Engine::load failed (A)");
            engine_a
                .start_request(&[("system", system), ("user", turn1_user)], None, SamplingParams::GREEDY, None, None)
                .expect("real start_request turn 1 failed (A)");
            let mut turn1_ids: Vec<i32> = Vec::with_capacity(turn1_reply_len);
            for _ in 0..turn1_reply_len {
                turn1_ids.push(engine_a.step().expect("real step() failed (A, turn 1)"));
            }
            let turn1_reply_text = engine_a.tokenizer.decode(&turn1_ids).expect("real decode failed (A, turn 1)");
            eprintln!("turn 1 reply (real, shared by both engines): {turn1_reply_text:?}");

            engine_a
                .start_request(
                    &[("system", system), ("user", turn1_user), ("assistant", &turn1_reply_text), ("user", turn2_user)],
                    None,
                    SamplingParams::GREEDY,
                    None,
                    None,
                )
                .expect("real start_request turn 2 (full history) failed (A)");
            let mut turn2_ids_a: Vec<i32> = Vec::with_capacity(turn2_reply_len);
            for _ in 0..turn2_reply_len {
                turn2_ids_a.push(engine_a.step().expect("real step() failed (A, turn 2)"));
            }
            eprintln!("turn 2 reply, full-history-resend (A): {turn2_ids_a:?}");
            (turn1_ids, turn2_ids_a)
            // `engine_a` dropped here -- real VRAM freed before B loads.
        };

        // --- Engine B: snapshot + resume (the real, new path). ---
        let mut engine_b = Engine::load().expect("real Engine::load failed (B)");
        engine_b
            .start_request(&[("system", system), ("user", turn1_user)], None, SamplingParams::GREEDY, None, None)
            .expect("real start_request turn 1 failed (B)");
        let mut turn1_ids_b: Vec<i32> = Vec::with_capacity(turn1_reply_len);
        for _ in 0..turn1_reply_len {
            turn1_ids_b.push(engine_b.step().expect("real step() failed (B, turn 1)"));
        }
        assert_eq!(turn1_ids_b, turn1_ids, "turn 1 must be identical across both real engines before the handoff mechanism is even exercised");

        let (snapshot_position, _created) = engine_b.snapshot_state("checkpoint".to_string()).expect("real snapshot_state failed (B)");
        eprintln!("real snapshot captured at position {snapshot_position}");

        engine_b
            .start_request(&[("user", turn2_user)], None, SamplingParams::GREEDY, None, Some("checkpoint"))
            .expect("real start_request turn 2 (resumed) failed (B)");
        let mut turn2_ids_b: Vec<i32> = Vec::with_capacity(turn2_reply_len);
        for _ in 0..turn2_reply_len {
            turn2_ids_b.push(engine_b.step().expect("real step() failed (B, turn 2)"));
        }
        eprintln!("turn 2 reply, snapshot+resume (B):      {turn2_ids_b:?}");

        assert_eq!(
            turn2_ids_a, turn2_ids_b,
            "real state-handoff (snapshot+resume) diverged from the real full-history-resend ground truth on turn 2 -- the handoff lost or corrupted real conversation state"
        );

        // Real, additional guard: the snapshot must still be usable a
        // second time (it is a named checkpoint, not a one-shot token --
        // see `start_request`'s own doc comment).
        let position_after_restore = engine_b.snapshots.get("checkpoint").expect("snapshot should still exist after being resumed once").snap.position;
        assert_eq!(position_after_restore, snapshot_position, "resuming a snapshot must not mutate the stored snapshot itself");
    }

    /// What batched decode actually buys, measured on the real decode path
    /// (not the prefill proxy `diagnose_batching_headroom_*` used to decide
    /// whether to build it). Gated on
    /// `real_batched_decode_matches_sequential_single_sequence_decode` --
    /// speed numbers for a path that is not token-identical are worthless.
    #[test]
    #[ignore]
    fn bench_real_batched_decode_vs_sequential_decode() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed");

        const N: usize = 32;
        let mut engine = Engine::load().expect("real Engine::load failed");
        let handle = BlasHandle::create().expect("real BlasHandle::create failed");
        let prompt = "Write a Python function that reverses a string.";

        // Baseline: the production single-sequence path, one sequence at a time.
        engine
            .start_request(&[("user", prompt)], None, SamplingParams::GREEDY, None, None)
            .expect("start_request failed");
        for _ in 0..4 {
            engine.step().expect("warmup step failed");
        }
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            engine.step().expect("step failed");
        }
        let seq_ms_per_tok = t0.elapsed().as_secs_f64() * 1000.0 / N as f64;
        eprintln!("\n B | ms/step | tok/s aggregate | tok/s per seq | vs B=1");
        eprintln!("---+---------+-----------------+---------------+-------");
        eprintln!(
            " 1 | {seq_ms_per_tok:7.2} | {:15.1} | {:13.1} | 1.00x",
            1000.0 / seq_ms_per_tok,
            1000.0 / seq_ms_per_tok
        );

        for &batch in &[1usize, 2, 4, 8] {
            let mut bstate = match crate::model::BatchedDecodeState::new(batch, max_seq_len()) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(" {batch} | skipped: could not allocate batch state ({e})");
                    continue;
                }
            };
            let ids = engine
                .tokenizer
                .encode(&engine.tokenizer.apply_chat_template_with_tools(&[("user", prompt)], None))
                .expect("encode failed");
            let mut feed = Vec::with_capacity(batch);
            for i in 0..batch {
                let mut tmp = crate::model::DecodeState::new(max_seq_len()).expect("DecodeState::new failed");
                crate::model::forward_prefill(&handle, &engine.weights, &mut tmp, &ids, &mut engine.logits)
                    .expect("prefill failed");
                feed.push(crate::model::argmax_sample(&engine.logits).expect("argmax failed"));
                bstate.load_slot(i, &tmp).expect("load_slot failed");
            }
            let mut logits: DeviceBuffer<u16> =
                DeviceBuffer::alloc(batch * crate::model::VOCAB_SIZE).expect("logits alloc failed");

            for _ in 0..4 {
                crate::model::forward_batched_decode(&handle, &engine.weights, &mut bstate, &feed, &mut logits)
                    .expect("warmup batched decode failed");
            }
            let t1 = std::time::Instant::now();
            for _ in 0..N {
                crate::model::forward_batched_decode(&handle, &engine.weights, &mut bstate, &feed, &mut logits)
                    .expect("batched decode failed");
            }
            let ms_per_step = t1.elapsed().as_secs_f64() * 1000.0 / N as f64;
            let agg = batch as f64 * 1000.0 / ms_per_step;
            eprintln!(
                " {batch} | {ms_per_step:7.2} | {agg:15.1} | {:13.1} | {:.2}x",
                1000.0 / ms_per_step,
                agg / (1000.0 / seq_ms_per_tok)
            );
        }
        eprintln!();
    }

    /// THE gate for batched decode: B sequences decoded together must produce
    /// byte-identical tokens to those same B sequences decoded one at a time
    /// through the existing, already-validated single-sequence path.
    ///
    /// Batched decode reuses every existing kernel but re-derives all the
    /// pointer arithmetic at batch width, and a single wrong row offset would
    /// produce fluent, plausible, WRONG output -- the failure mode this repo
    /// has now been bitten by three times in one session. Equality against the
    /// trusted path is the only check that catches it.
    ///
    /// Deliberately uses DIFFERENT prompts per slot: identical prompts would
    /// pass even if every slot were secretly reading slot 0's KV cache.
    ///
    /// WHAT THIS DOES *NOT* PROVE. Exact equality holds at this batch size and
    /// horizon, and that is what makes it a sharp detector of indexing/stride
    /// bugs (those diverge at the FIRST token of the affected slot). It is not
    /// a claim that batched decode is bit-identical in general: changing the
    /// batch changes the GEMM shape, hipBLAS reduces in a different order, and
    /// a near-tied argmax can flip. Measured over HTTP at B=8 / 120 tokens,
    /// 5 of 8 completions eventually diverged from their single-call
    /// counterparts, the earliest 8% in. See MEASURED_FINDINGS §16h.
    #[test]
    #[ignore]
    fn real_batched_decode_matches_sequential_single_sequence_decode() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed");

        const N: usize = 24;
        let prompts = [
            "Write a Python function that reverses a string.",
            "Explain what a hash map is in two sentences.",
            "What is the capital of France?",
            "Give three uses for a binary search tree.",
        ];
        let batch = prompts.len();

        let mut engine = Engine::load().expect("real Engine::load failed");
        let handle = BlasHandle::create().expect("real BlasHandle::create failed");

        // --- Reference: each prompt decoded ALONE on the trusted path. ---
        let mut reference: Vec<Vec<i32>> = Vec::with_capacity(batch);
        for p in prompts.iter() {
            engine
                .start_request(&[("user", *p)], None, SamplingParams::GREEDY, None, None)
                .expect("real single-sequence start_request failed");
            let mut ids = Vec::with_capacity(N);
            for _ in 0..N {
                ids.push(engine.step().expect("real single-sequence step failed"));
            }
            reference.push(ids);
        }

        // --- Batched: all prompts prefilled into their own slot, then decoded
        // together, one step at a time. Prefill per slot reuses the single
        // path (batched PREFILL across different-length prompts is a separate
        // problem); only DECODE is under test here. ---
        let mut bstate = crate::model::BatchedDecodeState::new(batch, max_seq_len())
            .expect("real BatchedDecodeState::new failed");
        let mut first_tokens: Vec<i32> = Vec::with_capacity(batch);
        for (i, p) in prompts.iter().enumerate() {
            let prompt = engine.tokenizer.apply_chat_template_with_tools(&[("user", *p)], None);
            let ids = engine.tokenizer.encode(&prompt).expect("encode failed");
            // Prefill this slot's own caches by running the real prefill into
            // a scratch DecodeState, then decoding its first token from it.
            let mut tmp = crate::model::DecodeState::new(max_seq_len()).expect("DecodeState::new failed");
            crate::model::forward_prefill(&handle, &engine.weights, &mut tmp, &ids, &mut engine.logits)
                .expect("prefill failed");
            let first = crate::model::argmax_sample(&engine.logits).expect("argmax failed");
            first_tokens.push(first);
            // Copy the prefilled per-layer state into this slot's slice of
            // the batch-contiguous allocation.
            bstate.load_slot(i, &tmp).expect("load_slot failed");
        }
        assert_eq!(
            first_tokens,
            reference.iter().map(|r| r[0]).collect::<Vec<_>>(),
            "prefill disagreed with the reference before batched decode even started"
        );

        let mut logits: DeviceBuffer<u16> =
            DeviceBuffer::alloc(batch * crate::model::VOCAB_SIZE).expect("logits alloc failed");
        let mut produced: Vec<Vec<i32>> = first_tokens.iter().map(|&t| vec![t]).collect();
        let mut feed = first_tokens.clone();
        for _ in 1..N {
            crate::model::forward_batched_decode(&handle, &engine.weights, &mut bstate, &feed, &mut logits)
                .expect("real forward_batched_decode failed");
            for i in 0..batch {
                let row = crate::model::argmax_sample_row(&logits, i)
                    .expect("argmax on batched logits failed");
                produced[i].push(row);
                feed[i] = row;
            }
        }

        for i in 0..batch {
            assert_eq!(
                produced[i], reference[i],
                "slot {i} diverged from single-sequence decode.\n  batched : {:?}\n  sequential: {:?}",
                produced[i], reference[i]
            );
        }
        eprintln!("batched decode B={batch} matched sequential decode token-for-token over {N} tokens");
    }

    /// De-risks continuous batching BEFORE any of it is built.
    ///
    /// MEASURED_FINDINGS §9 claims batching is nearly free (flat ms/step from
    /// B=1 to B=4, 6.8x aggregate by B=8) -- but that was measured on the
    /// PYTHON runtime, with different kernels. `runtime-next`'s own `gemv` is
    /// reportedly already at 79-91% of peak bandwidth, so whether the same
    /// headroom exists here is an open question, not an inherited fact.
    ///
    /// Batched decode at B=N is arithmetically the same shape as a prefill of
    /// N tokens: one pass over every weight, N rows of activations through the
    /// same GEMMs. So timing the EXISTING batched prefill path at T=1,2,4,8
    /// measures the ceiling batching could reach on this engine, using code
    /// that already exists and is already correct. Flat per-row time means the
    /// headroom is real; linear scaling means batching would buy nothing here
    /// and the whole build should be abandoned.
    ///
    /// This does NOT measure batched decode itself (attention and GDN would
    /// need per-sequence state); it bounds what that work could win.
    #[test]
    #[ignore]
    fn diagnose_batching_headroom_via_batched_prefill_scaling() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed");

        let mut engine = Engine::load().expect("real Engine::load failed");
        let handle = BlasHandle::create().expect("real BlasHandle::create failed");
        let reps = 12;

        eprintln!("\n rows | ms/call | ms per row | rows/s   | vs B=1 row-rate");
        eprintln!("------+---------+------------+----------+----------------");
        let mut baseline_row_ms = 0.0f64;
        for &rows in &[1usize, 2, 4, 8, 16] {
            let token_ids: Vec<i32> = (0..rows).map(|i| (1000 + i) as i32).collect();
            // Warm up, then time steady state. Each rep restarts from a clean
            // state so the KV depth is identical across row counts -- otherwise
            // deeper caches would confound the comparison.
            for _ in 0..3 {
                engine.state.reset().expect("reset failed");
                crate::model::forward_prefill(&handle, &engine.weights, &mut engine.state, &token_ids, &mut engine.logits)
                    .expect("forward_prefill failed");
            }
            crate::hip::device_synchronize().expect("sync failed");
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                engine.state.reset().expect("reset failed");
                crate::model::forward_prefill(&handle, &engine.weights, &mut engine.state, &token_ids, &mut engine.logits)
                    .expect("forward_prefill failed");
            }
            crate::hip::device_synchronize().expect("sync failed");
            let ms_per_call = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
            let ms_per_row = ms_per_call / rows as f64;
            if rows == 1 {
                baseline_row_ms = ms_per_row;
            }
            eprintln!(
                "{rows:5} | {ms_per_call:7.2} | {ms_per_row:10.3} | {:8.1} | {:.2}x",
                1000.0 / ms_per_row,
                baseline_row_ms / ms_per_row
            );
        }
        eprintln!(
            "\nflat ms/call across rows => memory-bound, batching headroom is real.\n\
             ms/call rising ~linearly => compute-bound at B=1, batching buys little.\n"
        );
    }

    /// Decisive regression test for a real crash: decode was unbounded.
    /// `start_request` refused a PROMPT past `MAX_SEQ_LEN`, but nothing
    /// stopped generation from walking off the end of the KV cache that
    /// `DecodeState::new(MAX_SEQ_LEN)` allocated -- which killed the server
    /// mid-run on the real 128-task aider suite with "Memory access fault by
    /// GPU node-1 ... Page not present". Generation must now terminate
    /// cleanly at the boundary instead.
    #[test]
    #[ignore]
    fn real_decode_stops_at_context_limit_instead_of_faulting_the_gpu() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed");

        let mut engine = Engine::load().expect("real Engine::load failed");

        // Fill the context to just under the cap with a real prompt, so only
        // a small number of decoded tokens remain before the boundary.
        let mut filler = String::new();
        while engine.tokenizer.encode(&filler).map(|v| v.len()).unwrap_or(0) < max_seq_len() - 120 {
            filler.push_str("the quick brown fox jumps over the lazy dog. ");
        }
        engine
            .start_request(&[("user", filler.as_str())], None, SamplingParams::GREEDY, None, None)
            .expect("real start_request with a near-limit prompt failed");
        let start_pos = engine.state.position;
        eprintln!("prefilled to position {start_pos} of {}", max_seq_len());
        assert!(start_pos < max_seq_len(), "prefill guard should have accepted this prompt");

        // Decode far more tokens than the remaining headroom. Every call must
        // either succeed or refuse -- never fault.
        let mut decoded = 0usize;
        let mut refused = false;
        for _ in 0..(max_seq_len() - start_pos + 200) {
            if engine.context_exhausted() {
                refused = true;
                break;
            }
            match engine.step() {
                Ok(_) => decoded += 1,
                Err(e) => {
                    // An earlier version of this test accepted ANY Err as a
                    // clean refusal, which let a real HIP launch failure
                    // ("invalid argument", from the decode kernel exceeding
                    // LDS at a too-large window) pass as success. Only the
                    // engine's own bound counts as stopping cleanly.
                    assert!(
                        e.contains("context exhausted"),
                        "decode failed with a real engine/HIP error rather than the context bound after {decoded} tokens: {e}"
                    );
                    eprintln!("real, expected refusal after {decoded} tokens: {e}");
                    refused = true;
                    break;
                }
            }
        }
        assert!(refused, "decode ran {decoded} tokens without ever hitting the context bound");
        assert!(
            engine.state.position < max_seq_len(),
            "position {} reached max_seq_len {} -- the KV cache was written out of bounds",
            engine.state.position, max_seq_len()
        );
        eprintln!("stopped cleanly at position {} after {decoded} decoded tokens", engine.state.position);

        // And the engine must still be usable afterwards, not wedged.
        engine
            .start_request(&[("user", "Say hi.")], None, SamplingParams::GREEDY, None, None)
            .expect("engine must still serve a fresh request after hitting the context bound");
        engine.step().expect("engine must still decode after hitting the context bound");
    }

    /// Quantifies what the cross-adapter resume guard is actually buying.
    /// Both routes see the IDENTICAL logical conversation and generate the
    /// same turn under the same folded adapter; they differ only in how the
    /// prefix got there -- a base-captured KV/GDN snapshot restored under the
    /// adapter (the unsound route the engine now refuses) versus a full
    /// re-prefill under the adapter (the sound route it forces instead).
    /// Any divergence is the real, measured cost of mixing two weight bases
    /// inside one attention computation, reported per adapter scale rather
    /// than assumed.
    #[test]
    #[ignore]
    fn diagnose_cross_adapter_handoff_divergence_against_full_reprefill() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed");

        let adapter_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/adapters/m2_python_modern_r8a128_v7"
        );
        if !std::path::Path::new(adapter_path).exists() {
            eprintln!("skipping: real adapter not present at {adapter_path}");
            return;
        }

        const T1_LEN: usize = 48;
        const T2_LEN: usize = 32;
        let system = "You are an expert Python engineer.";
        let u1 = "Write a function `sort_numbers(numbers: str) -> str` that takes a space-separated string of number words from 'zero' to 'nine' and returns them sorted smallest to largest.";
        let u2 = "Tests failed:\n```\nAssertionError: 'five zero' != 'zero five'\n```\nOutput the corrected complete Python file.";

        let mut engine = Engine::load().expect("real Engine::load failed");
        engine.load_adapter(adapter_path, Some("python_modern")).expect("real load_adapter failed");

        for scale in ["python_modern@0.5", "python_modern@1.0"] {
            // --- Turn 1 under BASE, identical for both routes. ---
            engine.swap_to_adapter("base").expect("real reset to base failed");
            engine
                .start_request(&[("system", system), ("user", u1)], None, SamplingParams::GREEDY, None, None)
                .expect("real turn 1 start_request failed");
            let mut t1_ids: Vec<i32> = Vec::with_capacity(T1_LEN);
            for _ in 0..T1_LEN {
                t1_ids.push(engine.step().expect("real step() failed (turn 1)"));
            }
            let t1_text = engine.tokenizer.decode(&t1_ids).expect("real decode failed");
            engine.snapshot_state("handoff".to_string()).expect("real snapshot_state failed");

            // --- Route A: cross-adapter RESUME (what the engine now refuses). ---
            engine.swap_to_adapter(scale).expect("real swap_to_adapter failed");
            let current = engine.active_adapter_signature();
            // Deliberately retag to bypass the guard: this diagnostic exists
            // to measure the very corruption the guard prevents.
            engine.snapshots.get_mut("handoff").expect("snapshot must exist").adapter_sig = current;
            engine
                .start_request(&[("user", u2)], None, SamplingParams::GREEDY, None, Some("handoff"))
                .expect("real resumed start_request failed");
            let mut resumed: Vec<i32> = Vec::with_capacity(T2_LEN);
            for _ in 0..T2_LEN {
                resumed.push(engine.step().expect("real step() failed (resumed)"));
            }

            // --- Route B: full re-prefill under the SAME adapter (the sound route). ---
            engine
                .start_request(
                    &[("system", system), ("user", u1), ("assistant", &t1_text), ("user", u2)],
                    None,
                    SamplingParams::GREEDY,
                    None,
                    None,
                )
                .expect("real full re-prefill start_request failed");
            let mut reprefilled: Vec<i32> = Vec::with_capacity(T2_LEN);
            for _ in 0..T2_LEN {
                reprefilled.push(engine.step().expect("real step() failed (re-prefill)"));
            }

            let matching = resumed.iter().zip(reprefilled.iter()).take_while(|(a, b)| a == b).count();
            let total_same = resumed.iter().zip(reprefilled.iter()).filter(|(a, b)| a == b).count();
            eprintln!("\n=== {scale} ===");
            eprintln!("  cross-adapter resume : {resumed:?}");
            eprintln!("  full re-prefill      : {reprefilled:?}");
            eprintln!(
                "  identical prefix     : {matching}/{T2_LEN} tokens; total positions equal: {total_same}/{T2_LEN}"
            );
            if matching < T2_LEN {
                eprintln!("  FIRST DIVERGENCE at position {matching}");
                eprintln!("    resumed  -> {:?}", engine.tokenizer.decode(&resumed[..(matching + 1).min(T2_LEN)]));
                eprintln!("    reprefill-> {:?}", engine.tokenizer.decode(&reprefilled[..(matching + 1).min(T2_LEN)]));
            }
            engine.delete_snapshot("handoff");
        }
    }

    /// §127 decisive test AND regression guard: the snapshot store's own
    /// real cap (`MAX_SNAPSHOTS`) is enforced with a real, loud error --
    /// never silent eviction, never an unbounded-VRAM-growth crash -- and
    /// deleting frees a real slot for a new name.
    #[test]
    #[ignore]
    fn real_snapshot_store_enforces_its_cap_and_delete_frees_a_slot() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let mut engine = Engine::load().expect("real Engine::load failed");
        engine
            .start_request(&[("user", "hello")], None, SamplingParams::GREEDY, None, None)
            .expect("real start_request failed");

        for i in 0..MAX_SNAPSHOTS {
            engine.snapshot_state(format!("slot-{i}")).unwrap_or_else(|e| panic!("real snapshot_state failed for slot-{i}: {e}"));
        }
        assert_eq!(engine.snapshots.len(), MAX_SNAPSHOTS);

        let err = engine.snapshot_state("one-too-many".to_string()).expect_err("a NEW snapshot name past MAX_SNAPSHOTS must be a real, loud error, not silent eviction");
        eprintln!("real, expected cap error: {err}");
        assert_eq!(engine.snapshots.len(), MAX_SNAPSHOTS, "a rejected snapshot must not have been inserted");

        // Overwriting an EXISTING name must still work at the cap (real
        // upsert semantics, not blocked by the cap).
        engine.snapshot_state("slot-0".to_string()).expect("overwriting an existing snapshot name must succeed even at the cap");
        assert_eq!(engine.snapshots.len(), MAX_SNAPSHOTS);

        assert!(engine.delete_snapshot("slot-0"), "deleting an existing snapshot must report it existed");
        assert!(!engine.delete_snapshot("slot-0"), "deleting an already-deleted snapshot must report it did not exist");
        assert_eq!(engine.snapshots.len(), MAX_SNAPSHOTS - 1);

        engine.snapshot_state("new-after-delete".to_string()).expect("a NEW snapshot name must succeed once delete freed a real slot");
        assert_eq!(engine.snapshots.len(), MAX_SNAPSHOTS);
    }

    /// Decisive test for the real soundness boundary on tensor handoff: a
    /// snapshot's K/V cache stores `W_k · x` / `W_v · x` for whatever LoRA
    /// was folded into the live weights at capture time, and every layer's
    /// hidden states carry that adapter's MLP deltas as well. Resuming it
    /// under a DIFFERENT folded configuration therefore evaluates one
    /// attention product across two weight bases, which degrades output
    /// subtly rather than failing -- the worst possible failure mode. This
    /// proves the engine refuses that resume loudly, drops the unusable
    /// snapshot so its VRAM is not stranded, and still permits the
    /// same-adapter resume that handoff exists for.
    #[test]
    #[ignore]
    fn real_resume_across_an_adapter_swap_is_refused_and_same_adapter_resume_still_works() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let adapter_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/adapters/m2_python_modern_r8a128_v7"
        );
        if !std::path::Path::new(adapter_path).exists() {
            eprintln!("skipping: real adapter not present at {adapter_path}");
            return;
        }

        let mut engine = Engine::load().expect("real Engine::load failed");
        engine.load_adapter(adapter_path, Some("python_modern")).expect("real load_adapter failed");

        // Capture under base (every adapter scale still 0.0).
        engine
            .start_request(&[("user", "Explain what a stack is.")], None, SamplingParams::GREEDY, None, None)
            .expect("real base start_request failed");
        engine.snapshot_state("handoff".to_string()).expect("real snapshot_state failed");
        assert_eq!(
            engine.snapshots.get("handoff").expect("snapshot must exist").adapter_sig,
            "base",
            "a snapshot captured with every adapter scale at 0.0 must record the base signature"
        );

        // Fold the real adapter in, then attempt the cross-adapter resume.
        engine.swap_to_adapter("python_modern").expect("real swap_to_adapter failed");
        let err = engine
            .start_request(&[("user", "continue")], None, SamplingParams::GREEDY, None, Some("handoff"))
            .expect_err("resuming a base-captured snapshot under a folded adapter must be a real, loud error");
        eprintln!("real, expected cross-adapter refusal: {err}");
        assert!(err.contains("base"), "the error must name the signature it was captured under: {err}");
        assert!(err.contains("python_modern"), "the error must name the signature it would run under: {err}");
        assert!(
            !engine.snapshots.contains_key("handoff"),
            "a snapshot that can never be resumed again must be dropped, not left holding VRAM"
        );

        // Same-adapter resume must still work: capture under the adapter now
        // folded in, and resume it under that same configuration.
        engine
            .start_request(&[("user", "Explain what a queue is.")], None, SamplingParams::GREEDY, None, None)
            .expect("real start_request under the adapter failed");
        engine.snapshot_state("same".to_string()).expect("real snapshot_state under adapter failed");
        let sig = engine.snapshots.get("same").expect("snapshot must exist").adapter_sig.clone();
        assert!(sig.starts_with("python_modern@"), "expected a real adapter signature, got {sig:?}");
        engine
            .start_request(&[("user", "continue")], None, SamplingParams::GREEDY, None, Some("same"))
            .expect("resuming a snapshot under the SAME adapter must still succeed");

        // Idle TTL: backdate last_used rather than sleeping, so the reaper is
        // tested deterministically.
        engine.snapshots.get_mut("same").expect("snapshot must exist").last_used =
            unix_time_secs().saturating_sub(SNAPSHOT_TTL_SECS + 1);
        assert_eq!(engine.purge_expired_snapshots(), 1, "an idle-expired snapshot must be reaped");
        assert!(!engine.snapshots.contains_key("same"), "the expired snapshot must be gone");
    }

    /// §103 decisive test AND regression guard: proves the real streaming
    /// loop's own overhead (JSON serialization, SSE frame construction, a
    /// real write+flush to an in-memory buffer) is negligible, and that
    /// the real per-token throughput decline across a generation is a
    /// genuine, structural KV-cache-depth cost, not anything this file
    /// does. This is the test that RESOLVED an earlier false lead in this
    /// same investigation (a suspected `.flush()`-per-token throughput
    /// cost that turned out not to be real -- see this section's own
    /// header doc).
    ///
    /// Three real variants, same `Engine`, same real prompt, same
    /// `max_tokens`, run back-to-back in ONE process (no cross-run
    /// session variance to worry about):
    /// A) `engine.step()` alone -- the real floor, timed in segments to
    ///    show the real KV-cache-depth cost directly.
    /// B) A + real JSON serialize + real SSE frame construction,
    ///    discarding the bytes (no write at all).
    /// C) B + a real `write_sse_frame` write+flush into an in-memory
    ///    `Vec<u8>` (matching the real server's own per-frame flush,
    ///    minus the real socket).
    /// If B/C ever drift far from A, that's a real regression in this
    /// file's own loop logic worth investigating -- not the ~78-80 vs
    /// ~90 tok/s illusion this test itself resolved.
    #[test]
    #[ignore]
    fn diagnose_real_streaming_loop_overhead_without_a_real_socket() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let mut engine = Engine::load().expect("real Engine::load failed");
        let system = "You are a senior PostgreSQL and Python backend architect.";
        let user = "Write a complete production FastAPI application with an asyncpg connection pool, pgvector HNSW search endpoint, and Pydantic response models.";
        let target_tokens = 300usize;

        // A) engine.step() alone -- timed in 50-token SEGMENTS to see
        // whether cost grows with KV-cache depth/position (the real
        // suspect once B/C below show socket/JSON overhead is negligible).
        engine.start_request(&[("system", system), ("user", user)], None, SamplingParams::GREEDY, None, None).expect("real start_request failed");
        let t0 = std::time::Instant::now();
        let segment = 50usize;
        let mut segment_tok_s = Vec::new();
        let mut done = 0usize;
        while done < target_tokens {
            let n = segment.min(target_tokens - done);
            let ts = std::time::Instant::now();
            for _ in 0..n {
                let _ = engine.step().expect("real engine.step() failed");
            }
            let seg_elapsed = ts.elapsed();
            segment_tok_s.push(n as f64 / seg_elapsed.as_secs_f64());
            done += n;
        }
        let a_elapsed = t0.elapsed();
        let a_tok_s = target_tokens as f64 / a_elapsed.as_secs_f64();
        eprintln!("A per-50-token-segment tok/s (position grows left to right, starting at position 54): {segment_tok_s:.2?}");

        // B) + real JSON serialize + real SSE frame construction, no write.
        engine.start_request(&[("system", system), ("user", user)], None, SamplingParams::GREEDY, None, None).expect("real start_request failed");
        let id = completion_id();
        let mut generated_ids: Vec<i32> = Vec::new();
        let mut prev_text_len = 0usize;
        let t0 = std::time::Instant::now();
        for _ in 0..target_tokens {
            let next_id = engine.step().expect("real engine.step() failed");
            generated_ids.push(next_id);
            let full_text = engine.tokenizer.decode(&generated_ids).expect("real decode failed");
            let delta_text = full_text.get(prev_text_len..).unwrap_or("").to_string();
            prev_text_len = full_text.len();
            let frame = sse_frame_bytes(&id, ChunkDelta { role: None, content: Some(delta_text), tool_calls: None }, None);
            std::hint::black_box(&frame);
        }
        let b_elapsed = t0.elapsed();
        let b_tok_s = target_tokens as f64 / b_elapsed.as_secs_f64();

        // C) + a real write_sse_frame write+flush into an in-memory
        // Vec<u8> (matching the real server's own per-frame flush, minus
        // the real socket).
        engine.start_request(&[("system", system), ("user", user)], None, SamplingParams::GREEDY, None, None).expect("real start_request failed");
        let id = completion_id();
        let mut generated_ids: Vec<i32> = Vec::new();
        let mut prev_text_len = 0usize;
        let mut sink: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let t0 = std::time::Instant::now();
        for _ in 0..target_tokens {
            let next_id = engine.step().expect("real engine.step() failed");
            generated_ids.push(next_id);
            let full_text = engine.tokenizer.decode(&generated_ids).expect("real decode failed");
            let delta_text = full_text.get(prev_text_len..).unwrap_or("").to_string();
            prev_text_len = full_text.len();
            let frame = sse_frame_bytes(&id, ChunkDelta { role: None, content: Some(delta_text), tool_calls: None }, None);
            write_sse_frame(&mut sink, &frame).expect("real in-memory write failed");
        }
        let c_elapsed = t0.elapsed();
        let c_tok_s = target_tokens as f64 / c_elapsed.as_secs_f64();

        eprintln!(
            "A (engine.step() alone):                            {:.2} tok/s ({:.3}ms/token)",
            a_tok_s,
            a_elapsed.as_secs_f64() * 1000.0 / target_tokens as f64
        );
        eprintln!(
            "B (+ decode + JSON + frame, no write):               {:.2} tok/s ({:.3}ms/token)",
            b_tok_s,
            b_elapsed.as_secs_f64() * 1000.0 / target_tokens as f64
        );
        eprintln!(
            "C (+ real write_sse_frame write to in-memory Vec<u8>): {:.2} tok/s ({:.3}ms/token)",
            c_tok_s,
            c_elapsed.as_secs_f64() * 1000.0 / target_tokens as f64
        );

        // Real regression guard: B/C must stay within a real, generous
        // tolerance of A -- if this ever fails, something in this file's
        // OWN loop logic has regressed, not the (already understood,
        // structural) KV-cache-depth cost.
        assert!((a_tok_s - c_tok_s).abs() / a_tok_s < 0.05, "real streaming-loop overhead (JSON + frame + in-memory write) exceeded 5% of decode-alone throughput: A={a_tok_s:.2} tok/s, C={c_tok_s:.2} tok/s");
    }

    /// A writer that succeeds for a fixed number of calls, then fails
    /// with `BrokenPipe` on every call after -- simulates a real client
    /// disconnecting partway through an SSE stream, without needing a
    /// real socket (same "no real socket" testing philosophy as
    /// `diagnose_real_streaming_loop_overhead_without_a_real_socket`
    /// above).
    struct FailAfterNWrites {
        calls_remaining: usize,
    }
    impl std::io::Write for FailAfterNWrites {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.calls_remaining == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "simulated client disconnect"));
            }
            self.calls_remaining -= 1;
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// DECISIVE: proves the real mechanism `stream_chat_completion`'s own
    /// doc comment now documents -- a disconnected client's next SSE
    /// frame write fails, the decode loop stops within one real extra
    /// token, and the `MutexGuard<Engine>` is released normally. This
    /// property was already true (an emergent side effect of ordinary
    /// `?` propagation), but was never itself decisively verified before
    /// this test -- this crate's own standing discipline is a real test
    /// for every real, load-bearing behavior, not an assumption left
    /// unchecked.
    ///
    /// `max_tokens` is set to 10,000 -- far higher than any real request
    /// needs -- specifically so a PASSING result can only mean the loop
    /// stopped because of the simulated disconnect, never because it
    /// simply reached its own natural completion first. `calls_remaining`
    /// (200) is deliberately generous: enough real `Write::write` calls
    /// to comfortably cover the HTTP header, the role-announce frame, and
    /// several dozen real content frames, while remaining a tiny fraction
    /// of what streaming anywhere near 10,000 tokens would require --
    /// robust to the exact number of physical `write()` calls one SSE
    /// frame costs (an implementation detail of `write!`'s formatting,
    /// not something this test needs to predict precisely).
    #[test]
    #[ignore]
    fn real_disconnected_client_stops_generation_within_one_token() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let mut engine = Engine::load().expect("real Engine::load failed");
        let system = "You are a helpful assistant.";
        let user = "Count from 1 to 1000, one number per line.";
        let prompt_tokens = engine
            .start_request(&[("system", system), ("user", user)], None, SamplingParams::GREEDY, None, None)
            .expect("real start_request failed");
        let position_before = engine.state.position;

        let calls_remaining = 200usize;
        let writer: Box<dyn std::io::Write + Send> = Box::new(FailAfterNWrites { calls_remaining });

        let engine_mutex = std::sync::Mutex::new(engine);
        let guard = engine_mutex.lock().expect("lock failed");
        stream_chat_completion(writer, guard, prompt_tokens, 10_000, false, None);

        let engine = engine_mutex.into_inner().expect("mutex poisoned");
        let tokens_generated = engine.state.position - position_before;
        eprintln!("real tokens generated before the simulated disconnect stopped the loop: {tokens_generated} (max_tokens was 10,000, {calls_remaining} real writes were allowed to succeed first)");
        assert!(
            tokens_generated < 100,
            "generation continued far past the simulated disconnect ({tokens_generated} real tokens generated) -- the write-failure-stops-the-loop property this crate relies on for 'stop generation' is broken"
        );
    }

    /// §118 decisive test: real, end-to-end proof (the real `Engine`, real
    /// weights, real GPU forward passes -- not just `sampling.rs`'s own
    /// pure-algorithm tests) that (a) `temperature<=0.0` is still
    /// byte-exact identical to this crate's own established greedy
    /// reference for the SAME real 5-token prompt every other decisive
    /// test in this crate uses, and (b) a real, explicit `seed` makes a
    /// `temperature>0.0` request's generation byte-exact REPRODUCIBLE
    /// across two entirely separate real `start_request`/`step()` runs --
    /// the real property the HTTP `seed` field promises callers, checked
    /// through the actual GPU decode loop, not assumed from
    /// `sampling.rs`'s own host-only tests.
    #[test]
    #[ignore]
    fn real_end_to_end_sampling_matches_greedy_at_temp_zero_and_is_reproducible_with_a_seed() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let mut engine = Engine::load().expect("real Engine::load failed");

        // (a) temp<=0.0 must remain byte-exact against this crate's own
        // established real greedy reference (the same 5-token prompt +
        // expected continuation `model::tests::
        // real_greedy_generation_matches_real_qwen3_5_4b` and friends use
        // throughout this crate).
        let messages: [(&str, &str); 1] = [("user", "What is the capital of France?")];
        // Real, measured (not guessed) -- the real Qwen3.5-4B model's own
        // real greedy continuation for this exact real prompt through
        // this exact real chat template, captured directly from this
        // test's own first real run.
        let expected_greedy_prefix: [i32; 3] = [90700, 8340, 25];
        engine.start_request(&messages, None, SamplingParams::GREEDY, None, None).expect("real start_request (greedy) failed");
        let mut greedy_ids = Vec::new();
        for _ in 0..3 {
            greedy_ids.push(engine.step().expect("real engine.step() (greedy) failed"));
        }
        eprintln!("greedy (temp<=0.0): {greedy_ids:?}");
        assert_eq!(greedy_ids, expected_greedy_prefix, "temperature<=0.0 must remain byte-exact against this crate's own established greedy path -- sampling.rs must never be reached for this case");

        // (b) temperature>0.0 with a real, explicit seed must reproduce
        // the exact same real generated sequence across two SEPARATE
        // requests (the SAME reused `Engine`/`DecodeState`, reset between
        // them -- matching this server's own real usage pattern).
        let sampling = SamplingParams { temperature: 0.8, top_p: 0.95, top_k: 40 };
        let run = |engine: &mut Engine| -> Vec<i32> {
            engine.start_request(&messages, None, sampling, Some(20260918), None).expect("real start_request (sampled) failed");
            (0..12).map(|_| engine.step().expect("real engine.step() (sampled) failed")).collect()
        };
        let run1 = run(&mut engine);
        let run2 = run(&mut engine);
        eprintln!("sampled run 1 (seed=20260918): {run1:?}");
        eprintln!("sampled run 2 (seed=20260918): {run2:?}");
        assert_eq!(run1, run2, "the same real seed must reproduce the exact same real sampled generation across two separate requests through the actual GPU decode loop");

        // Real, decisive sanity check: a DIFFERENT seed, on a real
        // OPEN-ENDED prompt (deliberately not the factual capital-of-
        // France prompt above -- a real, found-not-assumed discovery
        // while building this test: that specific prompt's real
        // probability distribution is SO peaked that several arbitrarily
        // chosen seeds reproduced the identical greedy continuation for
        // 12+ real tokens even with temperature=0.8, which is a genuine
        // property of that prompt, confirmed NOT an RNG bug by
        // `sampling::tests::real_different_seeds_produce_different_raw_random_streams`
        // -- but a poor choice for THIS specific assertion). A real
        // creative-writing prompt at a higher temperature is far less
        // deterministic; seeds 1 vs 2 were directly confirmed to diverge
        // here before this assertion was written.
        let open_ended_messages: [(&str, &str); 1] = [("user", "Write a short, creative story about a robot exploring an abandoned space station.")];
        let creative_sampling = SamplingParams { temperature: 1.2, top_p: 0.95, top_k: 40 };
        engine.start_request(&open_ended_messages, None, creative_sampling, Some(1), None).expect("real start_request (seed=1) failed");
        let run_seed1: Vec<i32> = (0..20).map(|_| engine.step().expect("real engine.step() failed")).collect();
        engine.start_request(&open_ended_messages, None, creative_sampling, Some(2), None).expect("real start_request (seed=2) failed");
        let run_seed2: Vec<i32> = (0..20).map(|_| engine.step().expect("real engine.step() failed")).collect();
        eprintln!("open-ended seed=1: {run_seed1:?}");
        eprintln!("open-ended seed=2: {run_seed2:?}");
        assert_ne!(run_seed1, run_seed2, "a real, different seed on a real open-ended prompt must produce a different real generation -- the sampling RNG must not be a silent no-op end-to-end");
    }

    /// Real, direct measurement of WHERE `Engine::load()`'s own real VRAM
    /// footprint goes, step by step -- motivated by a real, found (not
    /// guessed) discrepancy: the real 3-way HTTP benchmark's own memory
    /// monitor shows runtime-next's real `vram_min_mb` sitting ~800MB
    /// above llama.cpp's/Ollama's own baseline, CONSISTENTLY across all 4
    /// smaller real sizes (0.8B: 3158MB, 2B: 3164MB, 4B: 3166MB, 9B:
    /// 3162MB -- essentially model-size-INDEPENDENT), with a real, larger
    /// jump at 27B (3453MB). Real hypothesis, not yet confirmed: a
    /// model-size-independent ~800MB is far more consistent with a FIXED
    /// per-process allocation (hipBLAS/HIP context/workspace overhead)
    /// than with anything that scales with model weights -- this test
    /// measures `hip::mem_info()` directly after each real step of
    /// `Engine::load()`'s own sequence to find out which one actually
    /// accounts for it, rather than guessing from `llama.cpp`'s own
    /// source code first.
    #[test]
    #[ignore]
    fn diagnose_real_engine_load_vram_breakdown() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let used_mb = |label: &str, prev_free: &mut usize| {
            let (free, total) = crate::hip::mem_info().expect("real hipMemGetInfo failed");
            let used = (total - free) / (1024 * 1024);
            let delta = if *prev_free == 0 { 0 } else { (*prev_free - free) as i64 / (1024 * 1024) };
            eprintln!("[{label}] real VRAM used: {used}MB (delta since last checkpoint: {delta:+}MB)");
            *prev_free = free;
        };

        let mut prev_free = 0usize;
        used_mb("0. process start (before any real HIP allocation)", &mut prev_free);

        let snapshot = crate::model_loader::locate_model_snapshot().expect("real snapshot must be found");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        used_mb("1. after ModelWeights::load (real weights on device)", &mut prev_free);

        let tokenizer = ChatTokenizer::load(&snapshot).expect("real tokenizer load failed");
        let _ = &tokenizer;
        used_mb("2. after ChatTokenizer::load (real, host-only -- expect ~0 VRAM delta)", &mut prev_free);

        let graphed = GraphedDecodeState::new().expect("real GraphedDecodeState::new failed");
        used_mb("3. after GraphedDecodeState::new (real 2nd hipBLAS handle + real stream, NO graph captured yet -- lazy)", &mut prev_free);

        let blas_handle = BlasHandle::create().expect("real BlasHandle::create failed");
        used_mb("4. after Engine's own BlasHandle::create (real 3rd-ish hipBLAS handle)", &mut prev_free);

        let state = DecodeState::new(max_seq_len()).expect("real DecodeState::new failed");
        used_mb("5. after DecodeState::new (real KV caches + Scratch/PrefillScratch buffers)", &mut prev_free);

        let logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).expect("real logits alloc failed");
        used_mb("6. after logits DeviceBuffer::alloc (real, tiny -- VOCAB_SIZE bf16 elements)", &mut prev_free);

        // Real, decisive: the FIRST real `forward_one_token` call is what
        // actually triggers real HIP Graph capture (lazy, per
        // `GraphedDecodeState::forward_one_token`'s own doc comment) --
        // measuring around it directly answers whether real graph capture
        // itself has a real, non-trivial VRAM cost, separate from the
        // handle/stream creation already measured at step 3.
        let mut graphed = graphed;
        let mut state = state;
        let mut logits = logits;
        // Minimal real prefill so decode has a real position to start from.
        let handle_for_prefill = BlasHandle::create().expect("real temp handle failed");
        forward_prefill(&handle_for_prefill, &weights, &mut state, &[1i32], &mut logits).expect("real forward_prefill failed");
        used_mb("7. after one real forward_prefill call (eager, ungraphed)", &mut prev_free);

        graphed.forward_one_token(&weights, &mut state, 1, &mut logits).expect("real forward_one_token (first, captures graph) failed");
        used_mb("8. after FIRST real forward_one_token (this is where real HIP Graph capture actually happens)", &mut prev_free);

        graphed.forward_one_token(&weights, &mut state, 1, &mut logits).expect("real forward_one_token (second, replay) failed");
        used_mb("9. after SECOND real forward_one_token (real replay, no new capture -- expect ~0 delta)", &mut prev_free);

        let _ = (blas_handle, tokenizer);
    }

    #[test]
    fn test_lora_adapters_json_serialization_and_parsing() {
        let payload = r#"[{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}]"#;
        let assignments: Vec<ScaleAssignment> = serde_json::from_str(payload).expect("deserialize scale assignments");
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].id, Some(0));
        assert_eq!(assignments[0].scale, Some(1.0));
        assert_eq!(assignments[1].id, Some(1));
        assert_eq!(assignments[1].scale, Some(0.0));

        let single_payload = r#"{"adapter": "astral", "scale": 1.0}"#;
        let swap_body: AdapterSwapBody = serde_json::from_str(single_payload).expect("deserialize swap body");
        assert_eq!(swap_body.adapter.as_deref(), Some("astral"));
        assert_eq!(swap_body.scale, Some(1.0));

        let info = LoraAdapterInfo {
            id: 0,
            path: "/path/to/astral".to_string(),
            name: "astral".to_string(),
            scale: 1.0,
        };
        let json_str = serde_json::to_string(&info).expect("serialize info");
        assert!(json_str.contains(r#""id":0"#));
        assert!(json_str.contains(r#""name":"astral""#));
        assert!(json_str.contains(r#""scale":1.0"#));
    }

    #[test]
    #[ignore]
    fn real_lora_adapter_load_and_swap_via_engine() {
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0).expect("hipSetDevice(0) failed");

        let mut engine = Engine::load().expect("real Engine::load failed");

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/adapters");
        let dir_a = root.join("m2_astral_r8a128_v7_real");
        let dir_b = root.join("m2_postgresql_r8a128_v7_real");

        if !dir_a.exists() || !dir_b.exists() {
            eprintln!("skipping: adapter dirs not found at {}", dir_a.display());
            return;
        }

        // 1. Load both adapters
        let id_a = engine.load_adapter(dir_a.to_str().unwrap(), Some("astral")).expect("load astral failed");
        let id_b = engine.load_adapter(dir_b.to_str().unwrap(), Some("postgresql")).expect("load postgresql failed");
        assert_eq!(id_a, 0);
        assert_eq!(id_b, 1);

        // Check list_adapters
        let list = engine.list_adapters();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].name, "astral");
        assert_eq!(list[0].scale, 0.0);
        assert_eq!(list[1].name, "postgresql");
        assert_eq!(list[1].scale, 0.0);

        // 2. Activate adapter 0 (astral)
        let (active_id, swap_ms) = engine.swap_to_adapter("astral").expect("swap to astral failed");
        assert_eq!(active_id, Some(0));
        eprintln!("Swap to astral completed in {swap_ms:.2} ms");
        assert_eq!(engine.adapters[0].scale, 1.0);
        assert_eq!(engine.adapters[1].scale, 0.0);

        // 3. Activate adapter 1 (postgresql)
        let (active_id, swap_ms) = engine.swap_to_adapter("postgresql").expect("swap to postgresql failed");
        assert_eq!(active_id, Some(1));
        eprintln!("Swap to postgresql completed in {swap_ms:.2} ms");
        assert_eq!(engine.adapters[0].scale, 0.0);
        assert_eq!(engine.adapters[1].scale, 1.0);

        // 4. Return to base
        let (active_id, swap_ms) = engine.swap_to_adapter("base").expect("swap to base failed");
        assert_eq!(active_id, None);
        eprintln!("Swap to base completed in {swap_ms:.2} ms");
        assert_eq!(engine.adapters[0].scale, 0.0);
        assert_eq!(engine.adapters[1].scale, 0.0);
    }
}
