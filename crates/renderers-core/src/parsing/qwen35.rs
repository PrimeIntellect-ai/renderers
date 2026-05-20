//! Qwen3.5 tool-call parser — XML-style tool calls with special-token thinking.
//!
//! Port of `renderers/parsing.py:parse_qwen35` + `_parse_xml_tool_calls`.
//!
//! Structural shape:
//!
//! ```text
//! <think>
//! ...reasoning text...
//! </think>
//!
//! ...content text...
//!
//! <tool_call>
//! <function=fn_name>
//! <parameter=key1>
//! value1
//! </parameter>
//! <parameter=key2>
//! value2
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! `<think>` and `</think>` are special tokens. Tool-call block contents are
//! parsed by regex on the decoded text — but the regex only runs inside the
//! bounded `<tool_call>...</tool_call>` span, never on the full completion.

use std::ops::Range;
use std::sync::LazyLock;

use regex::Regex;

use crate::parsing::{decode, find, find_from, strip_stop_tokens};
use crate::tokenizer::Tokenizer;
use crate::types::{ParsedResponse, ParsedToolCall, ToolArguments, ToolCallParseStatus};

static FUNCTION_NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<function=([^>]+)>").expect("function-name regex"));

static PARAMETER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<parameter=([^>]+)>\n?(.*?)\n?</parameter>").expect("parameter regex")
});

#[allow(clippy::too_many_arguments)]
pub fn parse_qwen35(
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    stop_ids: &[u32],
    think_id: u32,
    think_end_id: u32,
    tool_call_id: u32,
    tool_call_end_id: u32,
) -> ParsedResponse {
    let ids = strip_stop_tokens(token_ids, stop_ids);

    // ── Thinking: find </think> by token ID ─────────────────────────
    let mut reasoning: Option<String> = None;
    let mut parse_offset: usize = 0;
    let working_ids: Vec<u32>;
    let ids_after_think: &[u32] = match find(ids, think_end_id) {
        Some(think_end) => {
            // Filter out think_id tokens from the reasoning span so the
            // decoded text doesn't include the opening marker.
            let reasoning_ids: Vec<u32> = ids[..think_end]
                .iter()
                .copied()
                .filter(|&t| t != think_id)
                .collect();
            let txt = decode(tokenizer, &reasoning_ids).unwrap_or_default();
            reasoning = Some(txt.trim().to_string());
            parse_offset = think_end + 1;
            &ids[think_end + 1..]
        }
        None => {
            // <think> present but no </think> — truncated reasoning;
            // return early with reasoning-only response.
            if let Some(think_start) = find(ids, think_id) {
                let txt = decode(tokenizer, &ids[think_start + 1..]).unwrap_or_default();
                return ParsedResponse {
                    content: String::new(),
                    reasoning_content: Some(txt.trim().to_string()).filter(|s| !s.is_empty()),
                    tool_calls: Vec::new(),
                };
            }
            working_ids = ids.to_vec();
            &working_ids
        }
    };

    // ── Tool calls (token-bounded, regex-on-decoded-span) ───────────
    let (content_text, tool_calls) = match find(ids_after_think, tool_call_id) {
        Some(tc_start) => {
            let content = decode(tokenizer, &ids_after_think[..tc_start])
                .unwrap_or_default()
                .trim()
                .to_string();
            let tcs = parse_xml_tool_calls(
                tokenizer,
                &ids_after_think[tc_start..],
                tool_call_id,
                tool_call_end_id,
                parse_offset + tc_start,
            );
            (content, tcs)
        }
        None => {
            let content = decode(tokenizer, ids_after_think)
                .unwrap_or_default()
                .trim()
                .to_string();
            (content, Vec::new())
        }
    };

    ParsedResponse {
        content: content_text,
        reasoning_content: reasoning.filter(|s| !s.is_empty()),
        tool_calls,
    }
}

fn parse_xml_tool_calls(
    tokenizer: &Tokenizer,
    ids: &[u32],
    tc_id: u32,
    tc_end_id: u32,
    section_offset: usize,
) -> Vec<ParsedToolCall> {
    let mut out: Vec<ParsedToolCall> = Vec::new();
    let mut i = 0usize;

    while i < ids.len() {
        if ids[i] != tc_id {
            i += 1;
            continue;
        }
        let span_start = section_offset + i;

        let end = match find_from(ids, tc_end_id, i + 1) {
            Some(end) => end,
            None => {
                let raw = decode(tokenizer, &ids[i + 1..]).unwrap_or_default();
                out.push(ParsedToolCall {
                    raw,
                    token_span: Some(Range {
                        start: span_start,
                        end: section_offset + ids.len(),
                    }),
                    status: ToolCallParseStatus::UnclosedBlock,
                    ..Default::default()
                });
                break;
            }
        };

        let block_text = decode(tokenizer, &ids[i + 1..end]).unwrap_or_default();
        let span = Range {
            start: span_start,
            end: section_offset + end + 1,
        };

        let name_match = match FUNCTION_NAME_RE.captures(&block_text) {
            Some(c) => c,
            None => {
                out.push(ParsedToolCall {
                    raw: block_text,
                    token_span: Some(span),
                    status: ToolCallParseStatus::MalformedStructure,
                    ..Default::default()
                });
                i = end + 1;
                continue;
            }
        };
        let name = name_match
            .get(1)
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();

        let mut arguments = serde_json::Map::new();
        let mut any_json_fallback = false;
        for pm in PARAMETER_RE.captures_iter(&block_text) {
            let arg_name = pm.get(1).map(|m| m.as_str()).unwrap_or("").to_string();
            let arg_value = pm.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            match serde_json::from_str::<serde_json::Value>(arg_value) {
                Ok(v) => {
                    arguments.insert(arg_name, v);
                }
                Err(_) => {
                    arguments.insert(arg_name, serde_json::Value::String(arg_value.to_string()));
                    any_json_fallback = true;
                }
            }
        }

        let status = if any_json_fallback {
            ToolCallParseStatus::InvalidJson
        } else {
            ToolCallParseStatus::Ok
        };

        out.push(ParsedToolCall {
            raw: block_text,
            name: if name.is_empty() { None } else { Some(name) },
            arguments: Some(ToolArguments::Object(serde_json::Value::Object(arguments))),
            token_span: Some(span),
            status,
            ..Default::default()
        });
        i = end + 1;
    }

    out
}
