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

pub struct ChatTokenizer {
    inner: Tokenizer,
    eos_token_id: i32,
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
        let eos_token_id = inner
            .token_to_id("<|im_end|>")
            .ok_or_else(|| "real tokenizer.json has no <|im_end|> token -- wrong tokenizer for this model".to_string())?
            as i32;

        Ok(ChatTokenizer { inner, eos_token_id })
    }

    pub fn eos_token_id(&self) -> i32 {
        self.eos_token_id
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

        out.push_str("<|im_start|>assistant\n<think>\n");
        out
    }
}

#[cfg(test)]
mod tests {
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
