//! Real BPE tokenization for the real Qwen3.5-4B `tokenizer.json` -- §95's
//! prerequisite for a general-purpose HTTP server (the decisive tests in
//! `model.rs` all bypass this via a fixed, real, pre-tokenized prompt
//! obtained once from the real tokenizer; a server needs to do that itself,
//! for arbitrary real user input).
//!
//! Uses HuggingFace's own `tokenizers` crate (`Tokenizer::from_file`) against
//! the real `tokenizer.json` in this model's real snapshot directory -- not
//! a hand-rolled BPE implementation. Real, well-solved problem; link it,
//! same reasoning `model_loader.rs` already used for `safetensors` (see that
//! module's own doc comment).
//!
//! CHAT TEMPLATE SCOPE (real limitation, not silently assumed away): the
//! real `tokenizer_config.json` ships a full Jinja2 chat template (vision,
//! tool-calling, multi-turn reasoning-history-stripping -- 7756 chars).
//! `apply_chat_template` below hand-rolls ONLY the plain-text ChatML
//! reduction of that template (ordinary system/user/assistant messages, no
//! tools, no prior `<think>` content to strip) -- traced by hand against
//! the real template and confirmed correct for that case, but NOT a general
//! Jinja implementation. Safe for independent single-turn requests (what
//! this session's benchmark actually sends); NOT yet safe for a real
//! multi-turn interactive conversation carrying forward an assistant turn
//! that itself contained `<think>...</think>` content, or for tool use.

use std::path::Path;
use tokenizers::Tokenizer;

/// Plain text of a message's `content`: a string, or the concatenated `text`
/// parts of structured content; `null`/absent is empty.
pub fn message_text(m: &serde_json::Value) -> String {
    match m.get("content") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

pub struct ChatTokenizer {
    inner: Tokenizer,
    eos_token_id: i32,
    /// Every token id that ends a generation: the tokenizer's own `eos_token`
    /// plus every id in `generation_config.json`'s `eos_token_id` (int or list).
    /// MiMo-V2.6-Distill-Qwen-9B declares `[<|im_end|>, <|endoftext|>]`; stopping
    /// on only the first would run such a generation on to `max_tokens`.
    stop_ids: Vec<i32>,
    /// The model's OWN chat template (`chat_template::ChatTemplate`), rendered
    /// with HF parity. `None` only when the snapshot ships no template, in which
    /// case the built-in Qwen3.5 rendering below is the fallback.
    template: Option<crate::chat_template::ChatTemplate>,
}

impl ChatTokenizer {
    /// Loads the real `tokenizer.json` from `snapshot_dir` (the same
    /// directory `model_loader::locate_model_snapshot()` already finds the
    /// real weights in).
    pub fn load(snapshot_dir: &Path) -> Result<Self, String> {
        let tokenizer_path = snapshot_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("failed to load real tokenizer.json at {}: {e}", tokenizer_path.display()))?;

        // Real eos token for this model family, confirmed against the real
        // tokenizer_config.json's own `eos_token` field -- read from the
        // tokenizer's own vocab (not hand-copied as a numeric literal that
        // could silently drift from the real vocab).
        //
        // The EOS text now comes from the model's own tokenizer_config.json
        // (`eos_token`, a string or an AddedToken dict); `<|im_end|>` remains
        // the fallback for snapshots that do not declare one.
        let cfg: serde_json::Value = std::fs::read_to_string(snapshot_dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(serde_json::Value::Null);
        let eos_text = match cfg.get("eos_token") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Object(o)) => o.get("content").and_then(|v| v.as_str()).unwrap_or("<|im_end|>").to_string(),
            _ => "<|im_end|>".to_string(),
        };
        let eos_token_id = inner
            .token_to_id(&eos_text)
            .ok_or_else(|| format!("tokenizer.json has no {eos_text:?} token (the model's declared eos_token)"))?
            as i32;

        let mut stop_ids = vec![eos_token_id];
        let generation_cfg: serde_json::Value = std::fs::read_to_string(snapshot_dir.join("generation_config.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(serde_json::Value::Null);
        match generation_cfg.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => stop_ids.extend(n.as_i64().map(|v| v as i32)),
            Some(serde_json::Value::Array(a)) => stop_ids.extend(a.iter().filter_map(|v| v.as_i64()).map(|v| v as i32)),
            _ => {}
        }
        stop_ids.sort_unstable();
        stop_ids.dedup();

        let template = crate::chat_template::ChatTemplate::load(snapshot_dir)?;
        Ok(ChatTokenizer { inner, eos_token_id, stop_ids, template })
    }

    pub fn eos_token_id(&self) -> i32 {
        self.eos_token_id
    }

    /// True for every token the model declares as ending a generation.
    pub fn is_stop(&self, id: i32) -> bool {
        self.stop_ids.contains(&id)
    }

    pub fn stop_ids(&self) -> &[i32] {
        &self.stop_ids
    }

    /// Where the chat template in use came from (`None` = built-in fallback).
    pub fn template_source(&self) -> Option<&str> {
        self.template.as_ref().map(|t| t.source.as_str())
    }

    /// Renders a conversation for generation with the model's OWN template.
    ///
    /// `messages` are full OpenAI message objects -- `tool_calls`, `tool` results,
    /// `content: null` and structured content reach the template untouched.
    /// `kwargs` are the request's `chat_template_kwargs` (e.g. `enable_thinking`),
    /// passed through to the template exactly as HF/vLLM/SGLang do.
    pub fn render_chat(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<String, String> {
        if let Some(t) = &self.template {
            return t.render(messages, tools, true, kwargs);
        }
        // Fallback for snapshots without a template: the built-in Qwen3.5
        // rendering, which only understands (role, text) pairs.
        let pairs: Vec<(String, String)> = messages
            .iter()
            .map(|m| {
                let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user").to_string();
                (role, message_text(m))
            })
            .collect();
        let refs: Vec<(&str, &str)> = pairs.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
        let thinking = kwargs
            .and_then(|k| k.get("enable_thinking"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        Ok(self.apply_chat_template_full(&refs, tools, thinking))
    }

    /// Real BPE encode. `add_special_tokens=false`: the real ChatML special
    /// tokens (`<|im_start|>`, `<|im_end|>`) are already literal text in
    /// `apply_chat_template`'s output, encoded the same as any other text
    /// -- adding them a second time via the tokenizer's own special-token
    /// injection would double them up.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| format!("real tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }

    /// Real BPE decode. `skip_special_tokens=false`: callers doing
    /// incremental streaming decode need to see `<|im_end|>` etc. in the
    /// growing string to detect/strip it themselves at the response layer,
    /// same reasoning as leaving them un-skipped on encode.
    pub fn decode(&self, ids: &[i32]) -> Result<String, String> {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner
            .decode(&ids_u32, false)
            .map_err(|e| format!("real tokenizer decode failed: {e}"))
    }

    /// One real chat message: `role` is `"system"`, `"user"`, or
    /// `"assistant"`.
    ///
    /// Real reduction of the real Jinja chat template for this exact case
    /// (system message first if present, then alternating user/assistant,
    /// no tools, no prior reasoning content) -- see this module's own doc
    /// comment for the scope limitation. Ends with the real template's own
    /// generation-prompt tail: `<|im_start|>assistant\n<think>\n` (this
    /// model's real default is thinking-mode ON).
    pub fn apply_chat_template(&self, messages: &[(&str, &str)]) -> String {
        self.apply_chat_template_with_tools(messages, None)
    }

    /// One real chat message: `role` is `"system"`, `"user"`, or
    /// `"assistant"`. When `tools` are present, injects Qwen's standard tool-calling
    /// schema and guidance into the system prompt.
    pub fn apply_chat_template_with_tools(
        &self,
        messages: &[(&str, &str)],
        tools: Option<&[serde_json::Value]>,
    ) -> String {
        self.apply_chat_template_full(messages, tools, true)
    }

    /// The full template, including Qwen3.5's own `enable_thinking` switch.
    /// `false` reproduces the upstream template's `enable_thinking is false`
    /// branch exactly: the generation prompt ends `<think>\n\n</think>\n\n`,
    /// so the model answers directly instead of reasoning first. Used where the
    /// visible output IS the product (e.g. writing a rationale for a training
    /// record), and reasoning would only spend the token budget.
    pub fn apply_chat_template_full(
        &self,
        messages: &[(&str, &str)],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> String {
        let mut out = String::new();
        let tools_text = if let Some(ts) = tools {
            if !ts.is_empty() {
                let mut t_str = String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
                for tool in ts {
                    let tool_target = if let Some(func) = tool.get("function") {
                        func
                    } else {
                        tool
                    };
                    let tool_json = serde_json::to_string(tool_target).unwrap_or_default();
                    t_str.push_str(&tool_json);
                    t_str.push('\n');
                }
                t_str.push_str("</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call needed, do NOT include <tool_call> tags\n</IMPORTANT>");
                Some(t_str)
            } else {
                None
            }
        } else {
            None
        };

        let mut has_system = false;
        for &(role, content) in messages {
            if role == "system" {
                has_system = true;
                out.push_str("<|im_start|>system\n");
                if let Some(ref t) = tools_text {
                    out.push_str(t);
                    out.push_str("\n\n");
                }
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            } else {
                out.push_str("<|im_start|>");
                out.push_str(role);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
        }

        if !has_system {
            if let Some(ref t) = tools_text {
                let mut prefix = String::from("<|im_start|>system\n");
                prefix.push_str(t);
                prefix.push_str("<|im_end|>\n");
                out = format!("{prefix}{out}");
            }
        }

        if enable_thinking {
            out.push_str("<|im_start|>assistant\n<think>\n");
        } else {
            out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        }
        out
    }
}

#[cfg(test)]
mod tests {

    /// The engine's own entry point, end to end on the CPU: a Qwen3.5 fine-tune
    /// that ships its OWN template and TWO stop tokens (MiMo-V2.6-Distill-Qwen-9B)
    /// must be rendered with that template -- not the built-in Qwen3.5 one -- and
    /// stop on both ids. Skips (loudly) when the snapshot is not downloaded.
    #[test]
    fn a_fine_tune_is_rendered_with_its_own_template_and_stop_set() {
        let hub = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join(".cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Distill-Qwen-9B/snapshots");
        let Some(snap) = std::fs::read_dir(&hub).ok().and_then(|d| d.flatten().map(|e| e.path()).max()) else {
            eprintln!("SKIP: MiMo snapshot not downloaded");
            return;
        };
        let tok = ChatTokenizer::load(&snap).unwrap();
        assert!(tok.template_source().unwrap().ends_with("chat_template.jinja"), "must use the model's own template");
        assert_eq!(tok.stop_ids(), &[248044, 248046], "<|endoftext|> and <|im_end|>, from generation_config.json");
        assert!(tok.is_stop(248044) && tok.is_stop(248046));

        let goldens: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/chat_template_goldens.json")).unwrap(),
        )
        .unwrap();
        let case = goldens["models"]["mimo-v2.6-distill-qwen-9b"]["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "tool_round_trip_null_content")
            .unwrap();
        let rendered = tok
            .render_chat(case["messages"].as_array().unwrap(), case["tools"].as_array().map(|v| v.as_slice()), None)
            .unwrap();
        assert_eq!(rendered, case["expected"].as_str().unwrap());
    }

    use super::*;
    use crate::model_loader::locate_model_snapshot;

    /// Real correctness test: loads the real tokenizer, encodes then
    /// decodes a real sentence, and checks the round trip reproduces it
    /// (modulo the tokenizer's own real normalization, e.g. leading-space
    /// handling -- checked via `contains`, not brittle exact equality,
    /// since BPE round-trips are not always byte-identical for arbitrary
    /// whitespace).
    #[test]
    #[ignore]
    fn real_tokenizer_round_trips_a_real_sentence() {
        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let tok = ChatTokenizer::load(&snapshot).expect("real tokenizer.json failed to load");

        let text = "The capital of France is Paris.";
        let ids = tok.encode(text).expect("real encode failed");
        assert!(!ids.is_empty(), "real encode produced no tokens");
        let decoded = tok.decode(&ids).expect("real decode failed");
        assert!(
            decoded.contains("capital of France is Paris"),
            "round trip did not reproduce the real sentence: got {decoded:?}"
        );
    }

    /// §103 investigation (temporary, not a permanent decisive test):
    /// times `decode()` called on a REPEATEDLY GROWING id sequence (the
    /// real streaming-server usage pattern: `decode(&generated_ids)`
    /// called once per generated token, on the WHOLE sequence so far,
    /// not just the new token -- see `server.rs`'s own comment on why).
    /// Real question: is this a real, measurable per-step cost that
    /// compounds as the sequence grows (an O(n) redecode every step, O(n^2)
    /// total), or negligible next to a real ~12ms decode step.
    #[test]
    #[ignore]
    fn diagnose_real_decode_cost_on_growing_sequence() {
        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let tok = ChatTokenizer::load(&snapshot).expect("real tokenizer.json failed to load");

        // Real token ids, repeated to build a real ~350-token sequence
        // (matching the real benchmark's own generation length).
        let base_ids: Vec<i32> = tok.encode("The quick brown fox jumps over the lazy dog. ").expect("real encode failed");
        let mut generated_ids: Vec<i32> = Vec::new();
        let mut total_decode_time = std::time::Duration::ZERO;
        let mut per_step_ms = Vec::new();
        let target_len = 350usize;
        while generated_ids.len() < target_len {
            for &id in &base_ids {
                if generated_ids.len() >= target_len {
                    break;
                }
                generated_ids.push(id);
                let t0 = std::time::Instant::now();
                let _ = tok.decode(&generated_ids).expect("real decode failed");
                let elapsed = t0.elapsed();
                total_decode_time += elapsed;
                per_step_ms.push(elapsed.as_secs_f64() * 1000.0);
            }
        }
        let first_10_avg: f64 = per_step_ms[..10].iter().sum::<f64>() / 10.0;
        let last_10_avg: f64 = per_step_ms[per_step_ms.len() - 10..].iter().sum::<f64>() / 10.0;
        eprintln!(
            "real decode() cost over a growing {}-token sequence: total={:.2}ms, avg/step={:.4}ms, first-10-avg={:.4}ms, last-10-avg={:.4}ms",
            generated_ids.len(),
            total_decode_time.as_secs_f64() * 1000.0,
            total_decode_time.as_secs_f64() * 1000.0 / generated_ids.len() as f64,
            first_10_avg,
            last_10_avg
        );
    }

    /// Confirms the real prompt this crate's own decisive tests use
    /// ("The capital of France is") encodes to the EXACT SAME real token
    /// ids `model.rs`'s decisive tests hardcode (`[760, 6511, 314, 9338, 369]`,
    /// obtained independently via `scratchpad/gen_reference_generation.py`
    /// -- see that test's own doc comment) -- a real, independent
    /// cross-check that this crate's own tokenizer integration agrees with
    /// the real reference process that established those ids in the first
    /// place.
    #[test]
    #[ignore]
    fn real_tokenizer_matches_the_decisive_tests_own_reference_prompt_ids() {
        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let tok = ChatTokenizer::load(&snapshot).expect("real tokenizer.json failed to load");

        let ids = tok.encode("The capital of France is").expect("real encode failed");
        assert_eq!(ids, vec![760, 6511, 314, 9338, 369]);
    }

    #[test]
    #[ignore]
    fn real_eos_token_id_is_real_im_end() {
        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let tok = ChatTokenizer::load(&snapshot).expect("real tokenizer.json failed to load");
        // Independently confirmed via encode: "<|im_end|>" with special
        // tokens enabled should decode back to exactly this single id.
        let ids = tok.inner.encode("<|im_end|>", true).unwrap();
        assert_eq!(ids.get_ids().len(), 1, "expected <|im_end|> to be a single real special token");
        assert_eq!(ids.get_ids()[0] as i32, tok.eos_token_id());
    }
}
