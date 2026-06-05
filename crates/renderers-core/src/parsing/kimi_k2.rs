//! Kimi K2 tool-call parser. Port of
//! `renderers/parsing.py:parse_kimi_k2` + `_parse_kimi_k2_tool_calls`.
//!
//! Structural shape:
//!
//! ```text
//! ...content with optional <think>...</think> text tags...
//! <|tool_calls_section_begin|>
//!   <|tool_call_begin|>{id}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
//!   ...
//! <|tool_calls_section_end|>
//! ```
//!
//! `{id}` is `functions.{name}:{index}`. The parser strips the
//! `functions.` prefix and `:index` suffix to recover the function name.

use std::ops::Range;

use crate::parsing::{decode, find, find_from, strip_stop_tokens};
use crate::tokenizer::Tokenizer;
use crate::types::{ParsedResponse, ParsedToolCall, ToolArguments, ToolCallParseStatus};

#[allow(clippy::too_many_arguments)]
pub fn parse_kimi_k2(
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    stop_ids: &[u32],
    tool_calls_section_begin_id: u32,
    tool_calls_section_end_id: u32,
    tool_call_begin_id: u32,
    tool_call_argument_begin_id: u32,
    tool_call_end_id: u32,
) -> ParsedResponse {
    let ids = strip_stop_tokens(token_ids, stop_ids);

    let (content_ids, tool_calls) = match find(ids, tool_calls_section_begin_id) {
        Some(section_start) => {
            let content = &ids[..section_start];
            let section_end =
                find_from(ids, tool_calls_section_end_id, section_start + 1).unwrap_or(ids.len());
            let section_ids = &ids[section_start + 1..section_end];
            let tcs = parse_kimi_k2_calls(
                tokenizer,
                section_ids,
                tool_call_begin_id,
                tool_call_argument_begin_id,
                tool_call_end_id,
                section_start + 1,
            );
            (content, tcs)
        }
        None => (ids, Vec::new()),
    };

    let text = decode(tokenizer, content_ids).unwrap_or_default();
    let (reasoning, content) = if let Some((before, after)) = text.split_once("</think>") {
        let raw = before.replacen("<think>", "", 1);
        let r = raw.trim_matches('\n').trim().to_string();
        let c = after.trim_matches('\n').to_string();
        (Some(r).filter(|s| !s.is_empty()), c)
    } else {
        if let Some(think_at) = text.find("<think>") {
            // Truncated thinking — no closing tag
            let raw = &text[think_at + "<think>".len()..];
            let r = raw.trim_matches('\n').trim().to_string();
            return ParsedResponse {
                content: String::new(),
                reasoning_content: Some(r).filter(|s| !s.is_empty()),
                tool_calls: Vec::new(),
            };
        }
        (None, text)
    };

    ParsedResponse {
        content: content.trim().to_string(),
        reasoning_content: reasoning,
        tool_calls,
    }
}

fn parse_kimi_k2_calls(
    tokenizer: &Tokenizer,
    ids: &[u32],
    tc_begin_id: u32,
    tc_arg_begin_id: u32,
    tc_end_id: u32,
    section_offset: usize,
) -> Vec<ParsedToolCall> {
    let mut out: Vec<ParsedToolCall> = Vec::new();
    let mut i = 0usize;

    while i < ids.len() {
        if ids[i] != tc_begin_id {
            i += 1;
            continue;
        }
        let Some(arg_begin) = find_from(ids, tc_arg_begin_id, i + 1) else {
            let raw = decode(tokenizer, &ids[i + 1..]).unwrap_or_default();
            out.push(ParsedToolCall {
                raw,
                token_span: Some(Range {
                    start: section_offset + i,
                    end: section_offset + ids.len(),
                }),
                status: ToolCallParseStatus::MalformedStructure,
                ..Default::default()
            });
            break;
        };

        let (tc_end, unclosed) = match find_from(ids, tc_end_id, arg_begin + 1) {
            Some(v) => (v, false),
            None => (ids.len(), true),
        };

        let raw_id = decode(tokenizer, &ids[i + 1..arg_begin])
            .unwrap_or_default()
            .trim()
            .to_string();
        let args_str = decode(tokenizer, &ids[arg_begin + 1..tc_end])
            .unwrap_or_default()
            .trim()
            .to_string();
        let block_text = decode(tokenizer, &ids[i + 1..tc_end]).unwrap_or_default();
        let span = Range {
            start: section_offset + i,
            end: section_offset + tc_end + usize::from(!unclosed),
        };

        // Extract function name from "functions.{name}:{index}"
        let name_part = raw_id.split(':').next().unwrap_or("");
        let func_name = if let Some((_, n)) = name_part.split_once('.') {
            n.to_string()
        } else {
            name_part.to_string()
        };

        let mut invalid_json = false;
        let arguments = if let Ok(v) = serde_json::from_str::<serde_json::Value>(&args_str) {
            ToolArguments::Object(v)
        } else {
            invalid_json = true;
            ToolArguments::Raw(args_str.clone())
        };

        let status = if unclosed {
            ToolCallParseStatus::UnclosedBlock
        } else if func_name.is_empty() {
            ToolCallParseStatus::MissingName
        } else if invalid_json {
            ToolCallParseStatus::InvalidJson
        } else {
            ToolCallParseStatus::Ok
        };

        out.push(ParsedToolCall {
            raw: block_text,
            name: if func_name.is_empty() {
                None
            } else {
                Some(func_name)
            },
            arguments: Some(arguments),
            token_span: Some(span),
            status,
            id: if raw_id.is_empty() {
                None
            } else {
                Some(raw_id)
            },
        });
        i = tc_end + 1;
        if unclosed {
            break;
        }
    }
    out
}
