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
use crate::model::{DecodeState, GraphedDecodeState, ModelWeights, VOCAB_SIZE, argmax_sample, forward_prefill};
use crate::sampling::{SamplingParams, make_rng, sample_from_logits};
use crate::tokenizer::ChatTokenizer;
use crate::hip::DeviceBuffer;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

const MODEL_ID: &str = "qwen3.5:4b-rust";
/// §95 scope decision: well under `attention.hip`'s real hardware ceiling
/// (`(head_dim=256 + max_seq_len + threads=256) * 4 <= 65536` bytes of LDS
/// per block => `max_seq_len` must stay under ~15872), generous for real
/// coding-task prompts+completions.
const MAX_SEQ_LEN: usize = 4096;
const DEFAULT_MAX_TOKENS: usize = 512;

fn unix_time_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
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
        let state = DecodeState::new(MAX_SEQ_LEN).map_err(|e| e.to_string())?;
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
        })
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
    fn start_request(&mut self, messages: &[(&str, &str)], sampling: SamplingParams, seed: Option<u64>) -> Result<usize, String> {
        self.state.reset().map_err(|e| e.to_string())?;
        self.sampling = sampling;
        self.rng = make_rng(seed);
        let prompt = self.tokenizer.apply_chat_template(messages);
        let prompt_ids = self.tokenizer.encode(&prompt)?;

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
    fn step(&mut self) -> Result<i32, String> {
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

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
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
}

#[derive(Serialize)]
struct ResponseChatMessage {
    role: &'static str,
    content: String,
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
    };
    let json = serde_json::to_string(&chunk).unwrap_or_default();
    format!("data: {json}\n\n").into_bytes()
}

/// Runs one real streaming chat completion to completion, writing and
/// flushing each SSE frame as it's produced -- real measurement
/// (`diagnose_real_streaming_loop_overhead_without_a_real_socket`) shows
/// this costs nothing worth trading TTFT for (see this section's own
/// header doc).
fn stream_chat_completion(mut writer: Box<dyn Write + Send>, mut engine: std::sync::MutexGuard<Engine>, max_tokens: usize) {
    let write_result = (|| -> std::io::Result<()> {
        write!(
            writer,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n"
        )?;

        let id = completion_id();
        let role_frame = sse_frame_bytes(&id, ChunkDelta { role: Some("assistant"), content: None }, None);
        write_sse_frame(&mut *writer, &role_frame)?;

        let eos = engine.tokenizer.eos_token_id();
        let mut generated_ids: Vec<i32> = Vec::new();
        let mut prev_text_len = 0usize;
        loop {
            if generated_ids.len() >= max_tokens {
                let frame = sse_frame_bytes(&id, ChunkDelta { role: None, content: None }, Some("length"));
                write_sse_frame(&mut *writer, &frame)?;
                break;
            }
            let next_id = match engine.step() {
                Ok(next_id) => next_id,
                Err(e) => return Err(std::io::Error::other(e)),
            };
            if next_id == eos {
                let frame = sse_frame_bytes(&id, ChunkDelta { role: None, content: None }, Some("stop"));
                write_sse_frame(&mut *writer, &frame)?;
                break;
            }
            generated_ids.push(next_id);
            let full_text = engine.tokenizer.decode(&generated_ids).map_err(std::io::Error::other)?;
            // Emit only the new suffix -- decoding the whole growing
            // sequence each step (not token-by-token) correctly handles
            // multi-token UTF-8/BPE merge boundaries.
            let delta_text = full_text.get(prev_text_len..).unwrap_or("").to_string();
            prev_text_len = full_text.len();
            if delta_text.is_empty() {
                continue;
            }
            let frame = sse_frame_bytes(&id, ChunkDelta { role: None, content: Some(delta_text) }, None);
            write_sse_frame(&mut *writer, &frame)?;
        }

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
    let messages: Vec<(&str, &str)> = req.messages.iter().map(|m| (m.role.as_str(), m.content.as_str())).collect();
    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

    let mut guard = match engine.lock() {
        Ok(g) => g,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("engine lock poisoned: {e}")})));
            return;
        }
    };
    let prompt_tokens = match guard.start_request(&messages, sampling, req.seed) {
        Ok(n) => n,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("prefill failed: {e}")})));
            return;
        }
    };

    if req.stream {
        // §103: bypasses `tiny_http`'s own `Response`/`respond()` path
        // entirely -- see `stream_chat_completion`'s own doc comment for
        // why (that path's chunked writer buffers 8KB with no flush).
        let writer = request.into_writer();
        stream_chat_completion(writer, guard, max_tokens);
        return;
    }

    // Non-streaming: run the full decode loop to completion, then return
    // one JSON response.
    let mut generated_ids: Vec<i32> = Vec::new();
    let eos = guard.tokenizer.eos_token_id();
    let mut finish_reason = "length";
    for _ in 0..max_tokens {
        let next_id = match guard.step() {
            Ok(id) => id,
            Err(e) => {
                let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("decode failed: {e}")})));
                return;
            }
        };
        if next_id == eos {
            finish_reason = "stop";
            break;
        }
        generated_ids.push(next_id);
    }
    let content = match guard.tokenizer.decode(&generated_ids) {
        Ok(t) => t,
        Err(e) => {
            let _ = request.respond(json_response(500, &serde_json::json!({"error": format!("detokenize failed: {e}")})));
            return;
        }
    };
    let response_body = ChatCompletionResponse {
        id: completion_id(),
        object: "chat.completion",
        created: unix_time_secs(),
        model: MODEL_ID,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ResponseChatMessage { role: "assistant", content },
            finish_reason,
        }],
        usage: UsageInfo {
            prompt_tokens,
            completion_tokens: generated_ids.len(),
            total_tokens: prompt_tokens + generated_ids.len(),
        },
    };
    let _ = request.respond(json_response(200, &response_body));
}

/// Loads real weights, starts the real HTTP server on `port`, and serves
/// forever. Blocking -- meant to be the only thing `main.rs` does once
/// server mode is selected.
pub fn run(port: u16) -> Result<(), String> {
    let engine = Engine::load()?;
    let engine = Mutex::new(engine);

    let server = Server::http(("0.0.0.0", port)).map_err(|e| format!("failed to bind port {port}: {e}"))?;
    eprintln!("[runtime-next] real HTTP server listening on 0.0.0.0:{port}");

    for request in server.incoming_requests() {
        let method = request.method().clone();
        let url = request.url().to_string();
        match (method, url.as_str()) {
            (Method::Get, "/health") => handle_health(request),
            (Method::Get, "/v1/models") => handle_models(request),
            (Method::Post, "/v1/chat/completions") => handle_chat_completions(request, &engine),
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
        engine.start_request(&[("system", system), ("user", user)], SamplingParams::GREEDY, None).expect("real start_request failed");
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
        engine.start_request(&[("system", system), ("user", user)], SamplingParams::GREEDY, None).expect("real start_request failed");
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
            let frame = sse_frame_bytes(&id, ChunkDelta { role: None, content: Some(delta_text) }, None);
            std::hint::black_box(&frame);
        }
        let b_elapsed = t0.elapsed();
        let b_tok_s = target_tokens as f64 / b_elapsed.as_secs_f64();

        // C) + a real write_sse_frame write+flush into an in-memory
        // Vec<u8> (matching the real server's own per-frame flush, minus
        // the real socket).
        engine.start_request(&[("system", system), ("user", user)], SamplingParams::GREEDY, None).expect("real start_request failed");
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
            let frame = sse_frame_bytes(&id, ChunkDelta { role: None, content: Some(delta_text) }, None);
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
        engine.start_request(&messages, SamplingParams::GREEDY, None).expect("real start_request (greedy) failed");
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
            engine.start_request(&messages, sampling, Some(20260918)).expect("real start_request (sampled) failed");
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
        engine.start_request(&open_ended_messages, creative_sampling, Some(1)).expect("real start_request (seed=1) failed");
        let run_seed1: Vec<i32> = (0..20).map(|_| engine.step().expect("real engine.step() failed")).collect();
        engine.start_request(&open_ended_messages, creative_sampling, Some(2)).expect("real start_request (seed=2) failed");
        let run_seed2: Vec<i32> = (0..20).map(|_| engine.step().expect("real engine.step() failed")).collect();
        eprintln!("open-ended seed=1: {run_seed1:?}");
        eprintln!("open-ended seed=2: {run_seed2:?}");
        assert_ne!(run_seed1, run_seed2, "a real, different seed on a real open-ended prompt must produce a different real generation -- the sampling RNG must not be a silent no-op end-to-end");
    }
}
