//! Real parser for the tool-call format Qwen3.5's OWN chat template asks for.
//!
//! Why this module exists, precisely: `tokenizer::apply_chat_template_with_tools`
//! reproduces the real upstream Qwen3.5 template, which instructs the model to
//! reply in an **XML** function-call form:
//!
//! ```text
//! <tool_call>
//! <function=example_function_name>
//! <parameter=example_parameter_1>
//! value_1
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! That is NOT the `{"name":..., "arguments":{...}}` JSON payload that
//! most OpenAI-compatible clients (and this repo's own `benchmarks/bfcl`
//! runner) assume lives inside `<tool_call>` tags. Running `json.loads` on a
//! real Qwen3.5 tool call therefore fails 100% of the time -- which is the
//! real, measured reason `scorecard_bfcl_0.8b_*.json` reported 0/10 with
//! `"tool_calls": []` on every single case across all three arms: the harness
//! silently discarded well-formed calls. This module is the fix on the SERVER
//! side, so any standard client gets real OpenAI `tool_calls` and never has to
//! know the wire format at all.
//!
//! Both forms are accepted (JSON first, then XML) because they are cheap to
//! distinguish and a LoRA fine-tuned on JSON-style tool data may well emit the
//! other one. Neither is guessed at: JSON is only accepted when it really
//! parses into an object carrying a name.

use serde_json::{Map, Value};

/// One real tool call recovered from generated text. `arguments` is always a
/// JSON **object** (possibly empty), never a string -- the string encoding
/// OpenAI's wire format wants is applied at serialization time by the server.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Everything before and including the LAST `</think>` is reasoning, not an
/// answer. Same rule this repo already settled on for grading generated code
/// (`benchmarks/humaneval/executor.py`, which takes `rsplit("</think>", 1)[1]`
/// after a real bug where scratch drafts inside the think block were graded
/// instead of the final answer). A model that rehearses a call inside its
/// reasoning must not have that rehearsal executed.
///
/// If stripping leaves nothing, the original text is returned -- a generation
/// truncated mid-think is better scanned whole than treated as empty.
fn strip_reasoning(content: &str) -> &str {
    match content.rsplit_once("</think>") {
        Some((_, after)) if !after.trim().is_empty() => after,
        _ => content,
    }
}

/// Finds the declared JSON-Schema type of one parameter, so `"5"` off the wire
/// can become the real number `5`. Accepts both the OpenAI envelope
/// (`{"type":"function","function":{...}}`) and a bare function object, exactly
/// as `apply_chat_template_with_tools` itself does -- the two must agree or the
/// model would be shown one shape and parsed under another.
fn schema_type_for<'a>(tools: Option<&'a [Value]>, fn_name: &str, param: &str) -> Option<&'a str> {
    let tools = tools?;
    for tool in tools {
        let target = tool.get("function").unwrap_or(tool);
        if target.get("name").and_then(Value::as_str) != Some(fn_name) {
            continue;
        }
        return target
            .get("parameters")?
            .get("properties")?
            .get(param)?
            .get("type")?
            .as_str();
    }
    None
}

/// Applies the declared type to a raw string captured from a `<parameter=...>`
/// block. Unknown or absent schema types deliberately stay STRINGS rather than
/// being guessed at: silently turning a zip code `"01234"` into the number
/// `1234` is a real corruption, and the caller's own schema is the only
/// authority on intent. The one exception is an explicitly typed field whose
/// text does not parse -- that also stays a string, so a malformed number
/// surfaces as a visible wrong value instead of being dropped.
fn coerce(raw: &str, ty: Option<&str>) -> Value {
    let trimmed = raw.trim();
    match ty {
        Some("integer") | Some("int") => trimmed
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::String(raw.to_string())),
        Some("number") | Some("float") | Some("double") => trimmed
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(Value::Number)
            .unwrap_or_else(|| Value::String(raw.to_string())),
        Some("boolean") | Some("bool") => match trimmed.to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::String(raw.to_string()),
        },
        Some("array") | Some("list") | Some("object") | Some("dict") => {
            serde_json::from_str::<Value>(trimmed).unwrap_or_else(|_| Value::String(raw.to_string()))
        }
        // "string", anything else, or no schema at all.
        _ => Value::String(raw.to_string()),
    }
}

/// Strips the single framing newline the template's own example puts around a
/// parameter value, WITHOUT touching interior whitespace. Deliberately not
/// `trim()`: a real code-editing tool (the reason a harness like OpenHands
/// passes multi-line parameters at all) carries significant leading
/// indentation on its first line, and `trim()` would silently corrupt it.
fn unframe(value: &str) -> &str {
    let v = value.strip_prefix('\n').or_else(|| value.strip_prefix("\r\n")).unwrap_or(value);
    v.strip_suffix('\n').map(|s| s.strip_suffix('\r').unwrap_or(s)).unwrap_or(v)
}

/// Parses ONE `<tool_call>` body. Tries the JSON form first (only accepted when
/// it genuinely parses to an object with a name), then the real XML form.
fn parse_one(body: &str, tools: Option<&[Value]>) -> Option<ParsedToolCall> {
    let trimmed = body.trim();

    // Form 1: `{"name": "f", "arguments": {...}}`.
    if trimmed.starts_with('{') {
        if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(trimmed) {
            if let Some(name) = obj.get("name").and_then(Value::as_str) {
                let arguments = obj
                    .get("arguments")
                    .or_else(|| obj.get("parameters"))
                    .cloned()
                    .filter(Value::is_object)
                    .unwrap_or_else(|| Value::Object(Map::new()));
                return Some(ParsedToolCall { name: name.to_string(), arguments });
            }
        }
    }

    // Form 2: the real Qwen3.5 XML form the chat template actually asks for.
    let after_fn = trimmed.split_once("<function=")?.1;
    let (name, rest) = after_fn.split_once('>')?;
    let name = name.trim();
    if name.is_empty() {
        return None;
    }

    let mut arguments = Map::new();
    let mut cursor = rest;
    while let Some((_, after_open)) = cursor.split_once("<parameter=") {
        let Some((key, after_key)) = after_open.split_once('>') else { break };
        // A parameter block with no closing tag (a truncated generation) is
        // dropped rather than guessed at, and parsing stops there.
        let Some((raw_value, remainder)) = after_key.split_once("</parameter>") else { break };
        let key = key.trim();
        if !key.is_empty() {
            let ty = schema_type_for(tools, name, key);
            arguments.insert(key.to_string(), coerce(unframe(raw_value), ty));
        }
        cursor = remainder;
    }

    Some(ParsedToolCall { name: name.to_string(), arguments: Value::Object(arguments) })
}

/// Recovers every real tool call from one generation.
///
/// Returns an empty vec when the model made no call, which is a genuinely
/// different outcome from "made a call we could not read" -- an unclosed or
/// unreadable `<tool_call>` block is skipped, never fabricated into a call.
pub fn parse_tool_calls(content: &str, tools: Option<&[Value]>) -> Vec<ParsedToolCall> {
    let body = strip_reasoning(content);
    let mut calls = Vec::new();
    let mut cursor = body;
    while let Some((_, after_open)) = cursor.split_once("<tool_call>") {
        let Some((inner, remainder)) = after_open.split_once("</tool_call>") else { break };
        if let Some(call) = parse_one(inner, tools) {
            calls.push(call);
        }
        cursor = remainder;
    }
    calls
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn triangle_tools() -> Vec<Value> {
        vec![json!({
            "type": "function",
            "function": {
                "name": "calculate_triangle_area",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                        "unit": {"type": "string"}
                    }
                }
            }
        })]
    }

    /// The exact shape `apps/runtime-next/src/tokenizer.rs` tells the model to
    /// emit, against the exact schema `benchmarks/bfcl` case `simple_0` sends.
    /// This is the case that really scored 0/10 before this module existed.
    #[test]
    fn real_qwen35_xml_tool_call_parses_with_schema_typed_arguments() {
        let content = "<think>\nI should use the area formula.\n</think>\n\
            I'll compute that.\n\
            <tool_call>\n<function=calculate_triangle_area>\n\
            <parameter=base>\n10\n</parameter>\n\
            <parameter=height>\n5\n</parameter>\n\
            </function>\n</tool_call>";
        let tools = triangle_tools();
        let calls = parse_tool_calls(content, Some(&tools));
        assert_eq!(calls.len(), 1, "exactly one real call must be recovered");
        assert_eq!(calls[0].name, "calculate_triangle_area");
        // Schema says integer, so these must be real JSON numbers -- BFCL's
        // ground truth compares against 10/5, not "10"/"5".
        assert_eq!(calls[0].arguments, json!({"base": 10, "height": 5}));
    }

    /// A call rehearsed inside the reasoning block must NOT be executed; only
    /// the real post-`</think>` answer counts.
    #[test]
    fn real_tool_call_inside_the_think_block_is_not_executed() {
        let content = "<think>\nMaybe <tool_call>\n<function=wrong_fn>\n</function>\n</tool_call>?\n\
            No, the other one.\n</think>\n\
            <tool_call>\n<function=right_fn>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(content, None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "right_fn", "the rehearsed call must be ignored");
    }

    /// Multi-line values keep their interior indentation exactly -- the real
    /// requirement for any code-editing tool call.
    #[test]
    fn real_multiline_parameter_preserves_interior_indentation() {
        let content = "<tool_call>\n<function=write_file>\n\
            <parameter=content>\ndef f():\n    return 1\n</parameter>\n\
            </function>\n</tool_call>";
        let calls = parse_tool_calls(content, None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["content"], json!("def f():\n    return 1"));
    }

    /// The JSON form other fine-tunes emit is accepted too.
    #[test]
    fn real_json_form_tool_call_is_also_accepted() {
        let content = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let calls = parse_tool_calls(content, None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, json!({"city": "Paris"}));
    }

    /// Two real parallel calls in one generation.
    #[test]
    fn real_parallel_tool_calls_are_all_recovered() {
        let content = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n\
            <tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(content, None);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    /// Plain prose with no call must produce no call -- the outcome that has
    /// to stay distinguishable from a parse failure.
    #[test]
    fn real_answer_without_a_tool_call_yields_no_calls() {
        assert!(parse_tool_calls("<think>\nno tool needed\n</think>\nThe answer is 25.", None).is_empty());
    }

    /// A truncated generation must never be completed by guesswork.
    #[test]
    fn real_truncated_tool_call_is_dropped_not_fabricated() {
        let content = "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>";
        assert!(parse_tool_calls(content, None).is_empty(), "no closing tag => no call");
    }

    /// Untyped and mistyped values stay strings rather than being invented.
    #[test]
    fn real_unknown_schema_keeps_values_as_strings() {
        let content = "<tool_call>\n<function=f>\n<parameter=zip>\n01234\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(content, None);
        assert_eq!(calls[0].arguments["zip"], json!("01234"), "no schema => no numeric guess");
    }

    /// Real booleans, floats and arrays coerce when the schema declares them.
    #[test]
    fn real_typed_scalars_and_arrays_coerce_from_schema() {
        let tools = vec![json!({
            "name": "f",
            "parameters": {"properties": {
                "flag": {"type": "boolean"},
                "ratio": {"type": "number"},
                "items": {"type": "array"}
            }}
        })];
        let content = "<tool_call>\n<function=f>\n\
            <parameter=flag>\ntrue\n</parameter>\n\
            <parameter=ratio>\n1.5\n</parameter>\n\
            <parameter=items>\n[1, 2]\n</parameter>\n\
            </function>\n</tool_call>";
        let calls = parse_tool_calls(content, Some(&tools));
        assert_eq!(calls[0].arguments, json!({"flag": true, "ratio": 1.5, "items": [1, 2]}));
    }
}
