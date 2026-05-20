//! DeepSeek V3 tool-call parser. Port of
//! `renderers/parsing.py:parse_deepseek_v3` + `_parse_deepseek_tool_calls`.
//!
//! Structural shape:
//!
//! ```text
//! ...content...
//! <think>...reasoning...</think>
//! <｜tool▁calls▁begin｜>
//!   <｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}
//!   ```json
//!   {args}
//!   ```<｜tool▁call▁end｜>
//! <｜tool▁calls▁end｜>
//! ```
//!
//! Thinking is **text tags** (not special tokens) — DeepSeek emits
//! `<think>...</think>` as decoded text. Tool calls are special-token
//! delimited. The fenced JSON inside is parsed with a small anchored regex.

use std::ops::Range;
use std::sync::LazyLock;

use regex::Regex;

use crate::parsing::{decode, find, find_from, strip_stop_tokens};
use crate::tokenizer::Tokenizer;
use crate::types::{ParsedResponse, ParsedToolCall, ToolArguments, ToolCallParseStatus};

static JSON_FENCE_RE: LazyLock<Regex> = LazyLock::new(|| {
    // Matches ```json\n<body>\n``` or ```\n<body>\n``` at the end of the string.
    Regex::new(r"(?s)^```(?:json)?\s*(.*?)\s*```$").expect("json-fence regex")
});

#[allow(clippy::too_many_arguments)]
pub fn parse_deepseek_v3(
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    stop_ids: &[u32],
    tool_calls_begin_id: u32,
    tool_calls_end_id: u32,
    tool_call_begin_id: u32,
    tool_call_end_id: u32,
    tool_sep_id: u32,
) -> ParsedResponse {
    let ids = strip_stop_tokens(token_ids, stop_ids);

    let (content_ids, tool_calls) = match find(ids, tool_calls_begin_id) {
        Some(section_start) => {
            let content = &ids[..section_start];
            let tcs = parse_deepseek_tool_calls(
                tokenizer,
                &ids[section_start..],
                tool_calls_begin_id,
                tool_calls_end_id,
                tool_call_begin_id,
                tool_call_end_id,
                tool_sep_id,
                section_start,
            );
            (content, tcs)
        }
        None => (ids, Vec::new()),
    };

    let text = decode(tokenizer, content_ids).unwrap_or_default();

    // Split out `<think>...</think>` from the decoded content. Plain text
    // tags here (no special tokens — that's the DeepSeek convention).
    let (reasoning, content) = match text.split_once("</think>") {
        Some((before, after)) => {
            let r = before
                .replace("<think>", "")
                .trim_matches('\n')
                .trim()
                .to_string();
            let c = after.trim_start_matches('\n').trim().to_string();
            (Some(r), c)
        }
        None => (None, text.trim().to_string()),
    };

    ParsedResponse {
        content,
        reasoning_content: reasoning.filter(|s| !s.is_empty()),
        tool_calls,
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_deepseek_tool_calls(
    tokenizer: &Tokenizer,
    ids: &[u32],
    tc_begin_id: u32,
    tc_end_id: u32,
    call_begin_id: u32,
    call_end_id: u32,
    sep_id: u32,
    section_offset: usize,
) -> Vec<ParsedToolCall> {
    let mut out: Vec<ParsedToolCall> = Vec::new();

    let Some(section_start) = find(ids, tc_begin_id) else {
        return out;
    };
    let section_end = find_from(ids, tc_end_id, section_start + 1).unwrap_or(ids.len());
    let inner_offset = section_offset + section_start + 1;
    let section_ids = &ids[section_start + 1..section_end];

    let mut i = 0usize;
    while i < section_ids.len() {
        if section_ids[i] != call_begin_id {
            i += 1;
            continue;
        }
        let (end, unclosed) = match find_from(section_ids, call_end_id, i + 1) {
            Some(end) => (end, false),
            None => (section_ids.len(), true),
        };
        let call_ids = &section_ids[i + 1..end];
        let block_text = decode(tokenizer, call_ids).unwrap_or_default();
        let span = Range {
            start: inner_offset + i,
            end: inner_offset + end + if unclosed { 0 } else { 1 },
        };

        let Some(sep_pos) = find(call_ids, sep_id) else {
            out.push(ParsedToolCall {
                raw: block_text,
                token_span: Some(span),
                status: ToolCallParseStatus::MalformedStructure,
                ..Default::default()
            });
            i = end + 1;
            continue;
        };

        let after_sep = decode(tokenizer, &call_ids[sep_pos + 1..])
            .unwrap_or_default()
            .trim()
            .to_string();

        let (name, args_str) = match after_sep.find('\n') {
            Some(nl) => {
                let n = after_sep[..nl].trim().to_string();
                let rest = after_sep[nl + 1..].trim();
                let args = match JSON_FENCE_RE.captures(rest) {
                    Some(c) => c
                        .get(1)
                        .map(|m| m.as_str().trim())
                        .unwrap_or("")
                        .to_string(),
                    None => rest.to_string(),
                };
                (n, args)
            }
            None => (after_sep.clone(), String::new()),
        };

        let mut invalid_json = false;
        let arguments = if args_str.is_empty() {
            ToolArguments::Object(serde_json::Value::Object(Default::default()))
        } else {
            match serde_json::from_str::<serde_json::Value>(&args_str) {
                Ok(v) => ToolArguments::Object(v),
                Err(_) => {
                    invalid_json = true;
                    ToolArguments::Raw(args_str.clone())
                }
            }
        };

        let status = if unclosed {
            ToolCallParseStatus::UnclosedBlock
        } else if name.is_empty() {
            ToolCallParseStatus::MissingName
        } else if invalid_json {
            ToolCallParseStatus::InvalidJson
        } else {
            ToolCallParseStatus::Ok
        };

        out.push(ParsedToolCall {
            raw: block_text,
            name: if name.is_empty() { None } else { Some(name) },
            arguments: Some(arguments),
            token_span: Some(span),
            status,
            ..Default::default()
        });
        i = end + 1;
        if unclosed {
            break;
        }
    }

    out
}
