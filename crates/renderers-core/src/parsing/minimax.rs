//! `MiniMax` M2 tool-call parser. Port of
//! `renderers/parsing.py:parse_minimax`.
//!
//! Structural shape:
//!
//! ```text
//! ...content...
//! <think>...reasoning...</think>     (special tokens)
//! <minimax:tool_call>
//!   <invoke name="fn">
//!     <parameter name="key1">value1</parameter>
//!     <parameter name="key2">value2</parameter>
//!   </invoke>
//!   ...possibly more <invoke> blocks in one wrapper...
//! </minimax:tool_call>
//! ```
//!
//! Thinking is special-token (`<think>` / `</think>`); the
//! tool-call block is bounded by special tokens but the inner
//! `<invoke>` / `<parameter>` structure is parsed by regex on the
//! decoded span.

use std::ops::Range;
use std::sync::LazyLock;

use regex::Regex;

use crate::parsing::{decode, find, find_from, strip_stop_tokens};
use crate::tokenizer::Tokenizer;
use crate::types::{ParsedResponse, ParsedToolCall, ToolArguments, ToolCallParseStatus};

static INVOKE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<invoke name="([^"]+)">(.*?)</invoke>"#).expect("invoke regex")
});
static PARAMETER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<parameter name="([^"]+)">(.*?)</parameter>"#).expect("parameter regex")
});

#[allow(clippy::too_many_arguments)]
pub fn parse_minimax(
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    stop_ids: &[u32],
    think_id: u32,
    think_end_id: u32,
    tool_call_id: u32,
    tool_call_end_id: u32,
) -> ParsedResponse {
    let stripped = strip_stop_tokens(token_ids, stop_ids);

    // Thinking
    let mut reasoning: Option<String> = None;
    let mut parse_offset = 0usize;
    let working: Vec<u32>;
    let ids: &[u32] = if let Some(think_end) = find(stripped, think_end_id) {
        let reasoning_ids: Vec<u32> = stripped[..think_end]
            .iter()
            .copied()
            .filter(|&t| t != think_id)
            .collect();
        let txt = decode(tokenizer, &reasoning_ids).unwrap_or_default();
        reasoning = Some(txt.trim().to_string()).filter(|s| !s.is_empty());
        parse_offset = think_end + 1;
        &stripped[think_end + 1..]
    } else {
        if let Some(think_start) = find(stripped, think_id) {
            let txt = decode(tokenizer, &stripped[think_start + 1..]).unwrap_or_default();
            return ParsedResponse {
                content: String::new(),
                reasoning_content: Some(txt.trim().to_string()).filter(|s| !s.is_empty()),
                tool_calls: Vec::new(),
            };
        }
        working = stripped.to_vec();
        &working
    };

    let mut tool_calls: Vec<ParsedToolCall> = Vec::new();
    let content_text = match find(ids, tool_call_id) {
        None => decode(tokenizer, ids)
            .unwrap_or_default()
            .trim()
            .to_string(),
        Some(tc_start) => {
            let content = decode(tokenizer, &ids[..tc_start])
                .unwrap_or_default()
                .trim()
                .to_string();
            let mut i = tc_start;
            while i < ids.len() {
                if ids[i] != tool_call_id {
                    i += 1;
                    continue;
                }
                let span_start = parse_offset + i;

                let Some(end) = find_from(ids, tool_call_end_id, i + 1) else {
                    let raw = decode(tokenizer, &ids[i + 1..]).unwrap_or_default();
                    tool_calls.push(ParsedToolCall {
                        raw,
                        token_span: Some(Range {
                            start: span_start,
                            end: parse_offset + ids.len(),
                        }),
                        status: ToolCallParseStatus::UnclosedBlock,
                        ..Default::default()
                    });
                    break;
                };
                let block_text = decode(tokenizer, &ids[i + 1..end]).unwrap_or_default();
                let span = Range {
                    start: span_start,
                    end: parse_offset + end + 1,
                };

                let invokes: Vec<_> = INVOKE_RE.captures_iter(&block_text).collect();
                if invokes.is_empty() {
                    tool_calls.push(ParsedToolCall {
                        raw: block_text,
                        token_span: Some(span),
                        status: ToolCallParseStatus::MalformedStructure,
                        ..Default::default()
                    });
                } else {
                    for inv in invokes {
                        let name = inv.get(1).map_or("", |m| m.as_str());
                        let body = inv.get(2).map_or("", |m| m.as_str());
                        let mut arguments = serde_json::Map::new();
                        let mut any_json_fallback = false;
                        for pm in PARAMETER_RE.captures_iter(body) {
                            let pname = pm.get(1).map_or("", |m| m.as_str());
                            let pval = pm.get(2).map_or("", |m| m.as_str().trim());
                            let v = if let Ok(v) = serde_json::from_str::<serde_json::Value>(pval) {
                                v
                            } else {
                                any_json_fallback = true;
                                serde_json::Value::String(pval.to_string())
                            };
                            arguments.insert(pname.to_string(), v);
                        }
                        let status = if any_json_fallback {
                            ToolCallParseStatus::InvalidJson
                        } else {
                            ToolCallParseStatus::Ok
                        };
                        tool_calls.push(ParsedToolCall {
                            raw: block_text.clone(),
                            name: Some(name.to_string()),
                            arguments: Some(ToolArguments::Object(serde_json::Value::Object(
                                arguments,
                            ))),
                            token_span: Some(span.clone()),
                            status,
                            ..Default::default()
                        });
                    }
                }
                i = end + 1;
            }
            content
        }
    };

    ParsedResponse {
        content: content_text,
        reasoning_content: reasoning,
        tool_calls,
    }
}
