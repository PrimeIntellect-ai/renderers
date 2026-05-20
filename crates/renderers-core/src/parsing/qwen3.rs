//! Qwen3 tool-call parser — Hermes-style JSON tool calls.
//!
//! Port of `renderers/parsing.py:parse_qwen3`. The structural shape is:
//!
//! ```text
//! ...content tokens...
//! <tool_call>
//! { "name": "fn", "arguments": { ... } }
//! </tool_call>
//! ...possibly more <tool_call> blocks...
//! ```
//!
//! Reasoning (`<think>...</think>`) is emitted as plain text by Qwen3
//! (not special tokens), so it falls out from the decoded content.

use crate::parsing::{decode, find, find_from, strip_stop_tokens};
use crate::tokenizer::Tokenizer;
use crate::types::{ParsedResponse, ParsedToolCall, ToolArguments, ToolCallParseStatus};

/// Parse Qwen3 completion tokens. `stop_ids` is consulted only to
/// truncate runaway content past EOS; the parser itself walks the
/// truncated prefix.
pub fn parse_qwen3(
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    stop_ids: &[u32],
    tool_call_id: u32,
    tool_call_end_id: u32,
) -> ParsedResponse {
    let ids = strip_stop_tokens(token_ids, stop_ids);

    let mut tool_calls: Vec<ParsedToolCall> = Vec::new();
    let (content_ids, _scanned) = match find(ids, tool_call_id) {
        Some(tc_start) => {
            let content = &ids[..tc_start];
            let mut i = tc_start;
            while i < ids.len() {
                if ids[i] == tool_call_id {
                    match find_from(ids, tool_call_end_id, i + 1) {
                        None => {
                            // No closing delim — runs to end of stripped ids.
                            let raw = decode(tokenizer, &ids[i + 1..])
                                .unwrap_or_default()
                                .trim()
                                .to_string();
                            tool_calls.push(ParsedToolCall {
                                raw,
                                token_span: Some(i..ids.len()),
                                status: ToolCallParseStatus::UnclosedBlock,
                                ..Default::default()
                            });
                            break;
                        }
                        Some(end) => {
                            let block = &ids[i + 1..end];
                            let tc_text = decode(tokenizer, block)
                                .unwrap_or_default()
                                .trim()
                                .to_string();
                            let span = i..(end + 1);
                            match serde_json::from_str::<serde_json::Value>(&tc_text) {
                                Err(_) => {
                                    tool_calls.push(ParsedToolCall {
                                        raw: tc_text,
                                        token_span: Some(span),
                                        status: ToolCallParseStatus::InvalidJson,
                                        ..Default::default()
                                    });
                                }
                                Ok(value) => {
                                    let (name, args) = extract_name_and_args(&value);
                                    if name.is_empty() {
                                        tool_calls.push(ParsedToolCall {
                                            raw: tc_text,
                                            name: None,
                                            arguments: Some(args),
                                            token_span: Some(span),
                                            status: ToolCallParseStatus::MissingName,
                                            ..Default::default()
                                        });
                                    } else {
                                        tool_calls.push(ParsedToolCall {
                                            raw: tc_text,
                                            name: Some(name),
                                            arguments: Some(args),
                                            token_span: Some(span),
                                            status: ToolCallParseStatus::Ok,
                                            ..Default::default()
                                        });
                                    }
                                }
                            }
                            i = end + 1;
                        }
                    }
                } else {
                    i += 1;
                }
            }
            (content, true)
        }
        None => (ids, false),
    };

    let text = decode(tokenizer, content_ids).unwrap_or_default();
    let (reasoning, content) = split_thinking(&text);

    ParsedResponse {
        content: content.trim().to_string(),
        reasoning_content: reasoning.filter(|s| !s.is_empty()),
        tool_calls,
    }
}

/// Pull `name` (string) and `arguments` (object or whatever the model
/// emitted) out of a parsed tool-call JSON value. Matches the Python
/// `parsed.get("name", "")` / `parsed.get("arguments", {})` semantics.
fn extract_name_and_args(value: &serde_json::Value) -> (String, ToolArguments) {
    let obj = match value.as_object() {
        Some(o) => o,
        None => return (String::new(), ToolArguments::default()),
    };
    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let args = match obj.get("arguments") {
        None => ToolArguments::default(),
        Some(serde_json::Value::String(s)) => ToolArguments::Raw(s.clone()),
        Some(v) => ToolArguments::Object(v.clone()),
    };
    (name, args)
}

/// Split a decoded text segment around `</think>`. Mirrors the inline
/// logic at `renderers/parsing.py` for Qwen3 (which has no `<think>` as
/// special token — reasoning lives in the decoded text).
fn split_thinking(text: &str) -> (Option<String>, String) {
    if let Some((before, after)) = text.split_once("</think>") {
        let reasoning = before
            .replace("<think>", "")
            .trim_matches('\n')
            .trim()
            .to_string();
        let content = after.trim_matches('\n').to_string();
        (Some(reasoning), content)
    } else {
        (None, text.to_string())
    }
}
