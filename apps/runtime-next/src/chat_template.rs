//! Renders a model's OWN chat template, the way Hugging Face `transformers` does.
//!
//! Why this exists: `tokenizer.rs` hand-rolled Qwen3.5's template, and a hand-rolled
//! copy only fits the one model it was copied from -- a Qwen3.5 fine-tune with its
//! own template (MiMo-V2.6-Distill-Qwen-9B: different tools preamble, no forced
//! `<think>`) would be prompted in a format it was never trained on. The copy was
//! also not byte-faithful for Qwen3.5 itself: HF's `tojson` is Python's
//! `json.dumps` (`", "` / `": "` separators, insertion-ordered keys) while the copy
//! used compact `serde_json`, so every tool-carrying prompt differed from what the
//! model saw in training.
//!
//! The contract is parity with `transformers.utils.chat_template_utils`
//! (`_compile_jinja_template` + `render_jinja_template`), pinned by golden tests
//! generated from real `transformers` (`tests/fixtures/chat_template_goldens.json`):
//!
//! * Jinja environment with `trim_blocks` and `lstrip_blocks`, loop controls,
//!   Python string/dict/list methods (minijinja-contrib `pycompat`).
//! * `tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)`
//!   reproducing `json.dumps` byte-for-byte.
//! * globals `raise_exception(msg)` and `strftime_now(fmt)`.
//! * context = special tokens (`bos_token`, `eos_token`, ...) overlaid by the
//!   caller's `chat_template_kwargs`, plus `messages`, `tools`,
//!   `add_generation_prompt` -- the same precedence HF uses.
//! * HF's `{% generation %}` block (an assistant-mask extension) renders its body
//!   unchanged; it is rewritten to an always-true `if` with the SAME whitespace
//!   control markers, so trimming around it is preserved exactly.

use std::path::Path;

use minijinja::value::Kwargs;
use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::{Map, Value as Json};

/// A compiled chat template plus the special tokens HF injects into its context.
pub struct ChatTemplate {
    env: Environment<'static>,
    has_tool_use_variant: bool,
    special_tokens: Map<String, Json>,
    /// Where the template came from, for logs and error messages.
    pub source: String,
}

impl ChatTemplate {
    /// Loads the template the model ships with. Returns `Ok(None)` when the
    /// snapshot has none (the caller then falls back to the built-in rendering).
    ///
    /// Lookup order matches `transformers`: a standalone `chat_template.jinja`
    /// wins over `tokenizer_config.json["chat_template"]`, which may be a string
    /// or a list of `{name, template}` entries (`default`, optional `tool_use`).
    pub fn load(snapshot_dir: &Path) -> Result<Option<Self>, String> {
        let cfg_path = snapshot_dir.join("tokenizer_config.json");
        let cfg: Json = match std::fs::read_to_string(&cfg_path) {
            Ok(text) => serde_json::from_str(&text).map_err(|e| format!("{}: {e}", cfg_path.display()))?,
            Err(_) => Json::Null,
        };

        let mut templates: Vec<(String, String)> = Vec::new();
        let jinja_path = snapshot_dir.join("chat_template.jinja");
        let source;
        if let Ok(text) = std::fs::read_to_string(&jinja_path) {
            templates.push(("default".into(), text));
            source = jinja_path.display().to_string();
        } else {
            match cfg.get("chat_template") {
                Some(Json::String(text)) => templates.push(("default".into(), text.clone())),
                Some(Json::Array(list)) => {
                    for entry in list {
                        if let (Some(name), Some(text)) =
                            (entry.get("name").and_then(Json::as_str), entry.get("template").and_then(Json::as_str))
                        {
                            templates.push((name.to_string(), text.to_string()));
                        }
                    }
                }
                _ => {}
            }
            source = format!("{}[chat_template]", cfg_path.display());
        }
        if templates.is_empty() {
            return Ok(None);
        }

        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson_filter);
        // Python semantics, not minijinja's: in Jinja2 `iterable` means
        // `iter(x)` succeeds, so None/numbers/bools are NOT iterable and
        // `tools is iterable and tools | length > 0` short-circuits on None.
        // minijinja iterates none as empty and then fails on `length`
        // (caught by the MiMo golden, whose template uses exactly that guard).
        env.add_test("iterable", |v: Value| {
            matches!(
                v.kind(),
                minijinja::value::ValueKind::String
                    | minijinja::value::ValueKind::Bytes
                    | minijinja::value::ValueKind::Seq
                    | minijinja::value::ValueKind::Map
                    | minijinja::value::ValueKind::Iterable
            )
        });
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        let mut has_tool_use_variant = false;
        for (name, text) in templates {
            has_tool_use_variant |= name == "tool_use";
            env.add_template_owned(name.clone(), rewrite_generation_tags(&text))
                .map_err(|e| format!("chat template {name:?} from {source} does not compile: {e:#}"))?;
        }
        if env.get_template("default").is_err() {
            return Err(format!("{source}: named templates present but no \"default\""));
        }

        // HF's `special_tokens_map`: string values, or AddedToken dicts reduced to
        // their content.
        let mut special_tokens = Map::new();
        for key in ["bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token"] {
            match cfg.get(key) {
                Some(Json::String(s)) => {
                    special_tokens.insert(key.into(), Json::String(s.clone()));
                }
                Some(Json::Object(o)) => {
                    if let Some(Json::String(s)) = o.get("content") {
                        special_tokens.insert(key.into(), Json::String(s.clone()));
                    }
                }
                _ => {}
            }
        }
        Ok(Some(Self { env, has_tool_use_variant, special_tokens, source }))
    }

    /// Renders one conversation. `messages` are full OpenAI-style message objects
    /// (tool calls, tool results and structured content pass through untouched);
    /// `kwargs` are the request's `chat_template_kwargs` and override special
    /// tokens on collision, exactly as in HF.
    pub fn render(
        &self,
        messages: &[Json],
        tools: Option<&[Json]>,
        add_generation_prompt: bool,
        kwargs: Option<&Map<String, Json>>,
    ) -> Result<String, String> {
        let mut ctx = self.special_tokens.clone();
        if let Some(kw) = kwargs {
            for (k, v) in kw {
                ctx.insert(k.clone(), v.clone());
            }
        }
        ctx.insert("messages".into(), Json::Array(messages.to_vec()));
        ctx.insert("tools".into(), tools.map(|t| Json::Array(t.to_vec())).unwrap_or(Json::Null));
        ctx.insert("add_generation_prompt".into(), Json::Bool(add_generation_prompt));

        let name = if tools.is_some() && self.has_tool_use_variant { "tool_use" } else { "default" };
        let template = self.env.get_template(name).map_err(|e| e.to_string())?;
        template
            .render(Value::from_serialize(&Json::Object(ctx)))
            .map_err(|e| format!("chat template ({}) failed to render: {e:#}", self.source))
    }
}

/// `{% generation %}` / `{% endgeneration %}` -> `{% if true %}` / `{% endif %}`,
/// preserving any `-` whitespace-control markers on the tags.
fn rewrite_generation_tags(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(start) = rest.find("{%") {
        out.push_str(&rest[..start]);
        let Some(end) = rest[start..].find("%}") else {
            out.push_str(&rest[start..]);
            return out;
        };
        let tag = &rest[start..start + end + 2];
        let inner = tag[2..tag.len() - 2].trim_start_matches('-').trim_end_matches('-').trim();
        let open = if tag.starts_with("{%-") { "{%-" } else { "{%" };
        let close = if tag.ends_with("-%}") { "-%}" } else { "%}" };
        match inner {
            "generation" => out.push_str(&format!("{open} if true {close}")),
            "endgeneration" => out.push_str(&format!("{open} endif {close}")),
            _ => out.push_str(tag),
        }
        rest = &rest[start + end + 2..];
    }
    out.push_str(rest);
    out
}

fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

/// `json.dumps(x, ensure_ascii=..., indent=..., separators=..., sort_keys=...)`.
fn tojson_filter(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<Value> = kwargs.get("indent")?;
    let separators: Option<Value> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;

    let json: Json = serde_json::to_value(&value)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
    let indent = match indent {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => match v.as_str() {
            Some(s) => Some(s.to_string()),
            None => {
                let n = i64::try_from(v).map_err(|_| Error::new(ErrorKind::InvalidOperation, "tojson: bad indent"))?;
                Some(" ".repeat(n.max(0) as usize))
            }
        },
    };
    // Python's defaults: (', ', ': ') without indent, (',', ': ') with one.
    let (item_sep, key_sep) = match separators {
        Some(v) if !v.is_none() => {
            let a: Vec<String> = v
                .try_iter()?
                .map(|x| x.as_str().map(str::to_string).unwrap_or_default())
                .collect();
            if a.len() != 2 {
                return Err(Error::new(ErrorKind::InvalidOperation, "tojson: separators must be a pair"));
            }
            (a[0].clone(), a[1].clone())
        }
        _ => (if indent.is_some() { ",".into() } else { ", ".into() }, ": ".into()),
    };
    let opts = DumpOpts {
        ensure_ascii: ensure_ascii.unwrap_or(false),
        indent,
        item_sep,
        key_sep,
        sort_keys: sort_keys.unwrap_or(false),
    };
    let mut out = String::new();
    dump(&json, &opts, 0, &mut out);
    Ok(Value::from(out))
}

struct DumpOpts {
    ensure_ascii: bool,
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
}

/// Serializes like CPython's `json` encoder (the pure-Python and C paths agree).
fn dump(v: &Json, o: &DumpOpts, depth: usize, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Number(n) => out.push_str(&python_number(n)),
        Json::String(s) => dump_str(s, o.ensure_ascii, out),
        Json::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(&o.item_sep);
                }
                newline_indent(o, depth + 1, out);
                dump(item, o, depth + 1, out);
            }
            newline_indent(o, depth, out);
            out.push(']');
        }
        Json::Object(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut entries: Vec<(&String, &Json)> = m.iter().collect();
            if o.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (k, val)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(&o.item_sep);
                }
                newline_indent(o, depth + 1, out);
                dump_str(k, o.ensure_ascii, out);
                out.push_str(&o.key_sep);
                dump(val, o, depth + 1, out);
            }
            newline_indent(o, depth, out);
            out.push('}');
        }
    }
}

fn newline_indent(o: &DumpOpts, depth: usize, out: &mut String) {
    if let Some(ind) = &o.indent {
        out.push('\n');
        for _ in 0..depth {
            out.push_str(ind);
        }
    }
}

/// Python prints floats with `repr` (`1.0`, `1e-05`, `1e+20`); integers plainly.
fn python_number(n: &serde_json::Number) -> String {
    if n.is_i64() || n.is_u64() {
        return n.to_string();
    }
    let f = n.as_f64().unwrap_or(f64::NAN);
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let exp = if f == 0.0 { 0 } else { f.abs().log10().floor() as i32 };
    // CPython's repr switches to exponent form when the exponent is < -4 or >= 16.
    if (-4..16).contains(&exp) {
        let s = format!("{f}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        // Rust `{:e}` gives `1e-5` / `1.5e20`; Python gives `1e-05` / `1.5e+20`.
        let s = format!("{f:e}");
        let (mant, e) = s.split_once('e').unwrap_or((&s, "0"));
        let e: i32 = e.parse().unwrap_or(0);
        format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    }
}

fn dump_str(s: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `datetime.now().strftime(fmt)` for the directives chat templates use
/// (`%d %b %Y`-style dates). Unsupported directives are an error, not a guess.
fn strftime_now(fmt: String) -> Result<Value, Error> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let offset = local_utc_offset_secs();
    let t = secs + offset;
    let days = t.div_euclid(86_400);
    let sod = t.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let weekday = (days + 4).rem_euclid(7) as usize; // 1970-01-01 was a Thursday
    const MON: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August",
                             "September", "October", "November", "December"];
    const DAY: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('B') => out.push_str(MON[(m - 1) as usize]),
            Some('b') => out.push_str(&MON[(m - 1) as usize][..3]),
            Some('A') => out.push_str(DAY[weekday]),
            Some('a') => out.push_str(&DAY[weekday][..3]),
            Some('H') => out.push_str(&format!("{:02}", sod / 3600)),
            Some('M') => out.push_str(&format!("{:02}", (sod / 60) % 60)),
            Some('S') => out.push_str(&format!("{:02}", sod % 60)),
            Some('%') => out.push('%'),
            other => {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("strftime_now: unsupported directive %{}", other.map(String::from).unwrap_or_default()),
                ))
            }
        }
    }
    Ok(Value::from(out))
}

/// Python renders `strftime_now` in LOCAL time. Reading the local offset needs
/// libc/tz bindings this crate does not carry, so the date is UTC unless
/// `TZ_OFFSET_SECS` is set. Templates use it for a date header only; this is the
/// one documented deviation from HF (dates are excluded from the golden tests).
fn local_utc_offset_secs() -> i64 {
    std::env::var("TZ_OFFSET_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
}

/// Howard Hinnant's days-from-civil inverse.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> DumpOpts {
        DumpOpts { ensure_ascii: false, indent: None, item_sep: ", ".into(), key_sep: ": ".into(), sort_keys: false }
    }

    #[test]
    fn dump_matches_python_json_dumps_defaults() {
        // json.dumps({"b": 1, "a": [1.0, None, True, "x\né"]}) with insertion order kept.
        let v: Json = serde_json::from_str(r#"{"b": 1, "a": [1.0, null, true, "x\né"]}"#).unwrap();
        let mut out = String::new();
        dump(&v, &opts(), 0, &mut out);
        assert_eq!(out, "{\"b\": 1, \"a\": [1.0, null, true, \"x\\n\u{e9}\"]}");
    }

    #[test]
    fn dump_ensure_ascii_escapes_like_python_including_surrogates() {
        let mut o = opts();
        o.ensure_ascii = true;
        let mut out = String::new();
        dump(&Json::String("é😀".into()), &o, 0, &mut out);
        assert_eq!(out, "\"\\u00e9\\ud83d\\ude00\"");
    }

    #[test]
    fn python_float_repr_edges() {
        let n = |f: f64| python_number(&serde_json::Number::from_f64(f).unwrap());
        assert_eq!(n(1.0), "1.0");
        assert_eq!(n(0.07), "0.07");
        assert_eq!(n(1e-5), "1e-05");
        assert_eq!(n(1.5e20), "1.5e+20");
    }

    #[test]
    fn generation_tags_keep_their_whitespace_control() {
        assert_eq!(rewrite_generation_tags("a{%- generation -%}b{% endgeneration %}c"), "a{%- if true -%}b{% endif %}c");
    }

    /// THE parity test. Every case in `tests/fixtures/chat_template_goldens.json`
    /// was rendered by real `transformers.apply_chat_template` from each model's own
    /// template (`gen_chat_template_goldens.py`); this renders the same inputs here
    /// and requires byte equality. Models whose snapshot is absent are reported and
    /// skipped, never silently counted as passing.
    #[test]
    fn chat_template_goldens_match_transformers() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/chat_template_goldens.json");
        let goldens: Json = serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).unwrap();
        let hub = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/huggingface/hub");
        let (mut checked, mut failures) = (0usize, Vec::new());
        for (model, spec) in goldens["models"].as_object().unwrap() {
            let snaps = hub.join(spec["repo_dir"].as_str().unwrap()).join("snapshots");
            let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|d| d.flatten().map(|e| e.path()).max()) else {
                eprintln!("SKIP {model}: snapshot not on disk");
                continue;
            };
            let tpl = ChatTemplate::load(&snap).unwrap().expect("model ships a template");
            for case in spec["cases"].as_array().unwrap() {
                let name = case["name"].as_str().unwrap();
                let messages = case["messages"].as_array().unwrap();
                let tools = case["tools"].as_array().map(|v| v.as_slice());
                let kwargs = case["kwargs"].as_object();
                let got = tpl.render(messages, tools, case["add_generation_prompt"].as_bool().unwrap(), kwargs);
                checked += 1;
                match (got, case.get("expected").and_then(Json::as_str)) {
                    (Ok(g), Some(want)) if g == want => {}
                    (Ok(g), Some(want)) => {
                        let at = g.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(g.len().min(want.len()));
                        let ctx = |s: &str| s.get(at.saturating_sub(40)..(at + 40).min(s.len())).unwrap_or("").to_string();
                        failures.push(format!("{model}/{name}: first diff at byte {at}\n  got : {:?}\n  want: {:?}", ctx(&g), ctx(want)));
                    }
                    (Err(e), Some(_)) => failures.push(format!("{model}/{name}: render error {e}")),
                    (Ok(_), None) => failures.push(format!("{model}/{name}: rendered, but transformers raised")),
                    (Err(_), None) => {}
                }
            }
        }
        assert!(checked > 0, "no golden could be checked -- are the snapshots downloaded?");
        assert!(failures.is_empty(), "{} of {checked} goldens differ:\n{}", failures.len(), failures.join("\n"));
        eprintln!("{checked} goldens byte-identical to transformers");
    }

    #[test]
    fn civil_from_days_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_725), (2026, 9, 29));
    }
}
