//! GLM tool-call parser — covers GLM-5 / GLM-5.1 / GLM-4.5.
//!
//! Port of `renderers/parsing.py:parse_glm` + `_parse_glm_tool_calls`.
//!
//! Structural shape:
//!
//! ```text
//! <|assistant|>...content...
//! <think>...reasoning...</think>
//! <tool_call>fn_name
//!   <arg_key>k1</arg_key><arg_value>v1</arg_value>
//!   <arg_key>k2</arg_key><arg_value>v2</arg_value>
//! </tool_call>
//! ```
//!
//! Thinking is special-token (`<think>` / `</think>`). Each argument is
//! a pair of special-token-delimited spans inside the tool-call block.
//! All scanning is token-id based — no decoded-text regex.

use std::ops::Range;

use crate::parsing::{decode, find, find_from, strip_stop_tokens};
use crate::tokenizer::Tokenizer;
use crate::types::{ParsedResponse, ParsedToolCall, ToolArguments, ToolCallParseStatus};

#[allow(clippy::too_many_arguments)]
pub fn parse_glm(
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    stop_ids: &[u32],
    think_id: u32,
    think_end_id: u32,
    tool_call_id: u32,
    tool_call_end_id: u32,
    arg_key_id: u32,
    arg_key_end_id: u32,
    arg_value_id: u32,
    arg_value_end_id: u32,
) -> ParsedResponse {
    let stripped = strip_stop_tokens(token_ids, stop_ids);

    // Thinking — find </think> by token id.
    let mut reasoning: Option<String> = None;
    let mut parse_offset = 0usize;
    let working_ids: Vec<u32>;
    let ids: &[u32] = match find(stripped, think_end_id) {
        Some(think_end) => {
            let reasoning_ids: Vec<u32> = stripped[..think_end]
                .iter()
                .copied()
                .filter(|&t| t != think_id)
                .collect();
            let txt = decode(tokenizer, &reasoning_ids).unwrap_or_default();
            reasoning = Some(txt.trim().to_string()).filter(|s| !s.is_empty());
            parse_offset = think_end + 1;
            &stripped[think_end + 1..]
        }
        None => {
            // Truncated reasoning — <think> without </think>
            if let Some(think_start) = find(stripped, think_id) {
                let txt = decode(tokenizer, &stripped[think_start + 1..]).unwrap_or_default();
                return ParsedResponse {
                    content: String::new(),
                    reasoning_content: Some(txt.trim().to_string()).filter(|s| !s.is_empty()),
                    tool_calls: Vec::new(),
                };
            }
            working_ids = stripped.to_vec();
            &working_ids
        }
    };

    let (content_text, tool_calls) = match find(ids, tool_call_id) {
        Some(tc_start) => {
            let content = decode(tokenizer, &ids[..tc_start])
                .unwrap_or_default()
                .trim()
                .to_string();
            let tcs = parse_glm_tool_calls(
                tokenizer,
                &ids[tc_start..],
                tool_call_id,
                tool_call_end_id,
                arg_key_id,
                arg_key_end_id,
                arg_value_id,
                arg_value_end_id,
                parse_offset + tc_start,
            );
            (content, tcs)
        }
        None => (
            decode(tokenizer, ids)
                .unwrap_or_default()
                .trim()
                .to_string(),
            Vec::new(),
        ),
    };

    ParsedResponse {
        content: content_text,
        reasoning_content: reasoning,
        tool_calls,
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_glm_tool_calls(
    tokenizer: &Tokenizer,
    ids: &[u32],
    tc_id: u32,
    tc_end_id: u32,
    ak_id: u32,
    ake_id: u32,
    av_id: u32,
    ave_id: u32,
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

        let block = &ids[i + 1..end];
        let block_text = decode(tokenizer, block).unwrap_or_default();
        let span = Range {
            start: span_start,
            end: section_offset + end + 1,
        };

        let first_ak = find(block, ak_id);
        let mut arguments = serde_json::Map::new();
        let mut any_json_fallback = false;
        let mut structure_broke = false;
        let name = match first_ak {
            None => decode(tokenizer, block)
                .unwrap_or_default()
                .trim()
                .to_string(),
            Some(first) => {
                let n = decode(tokenizer, &block[..first])
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                let mut j = first;
                while j < block.len() {
                    if block[j] != ak_id {
                        j += 1;
                        continue;
                    }
                    let Some(ake) = find_from(block, ake_id, j + 1) else {
                        structure_broke = true;
                        break;
                    };
                    let key = decode(tokenizer, &block[j + 1..ake])
                        .unwrap_or_default()
                        .trim()
                        .to_string();
                    let Some(av) = find_from(block, av_id, ake + 1) else {
                        structure_broke = true;
                        break;
                    };
                    let Some(ave) = find_from(block, ave_id, av + 1) else {
                        structure_broke = true;
                        break;
                    };
                    let val_text = decode(tokenizer, &block[av + 1..ave])
                        .unwrap_or_default()
                        .trim()
                        .to_string();
                    let val = match serde_json::from_str::<serde_json::Value>(&val_text) {
                        Ok(v) => v,
                        Err(_) => {
                            any_json_fallback = true;
                            serde_json::Value::String(val_text)
                        }
                    };
                    arguments.insert(key, val);
                    j = ave + 1;
                }
                n
            }
        };

        let status = if name.is_empty() {
            ToolCallParseStatus::MissingName
        } else if structure_broke {
            ToolCallParseStatus::MalformedStructure
        } else if any_json_fallback {
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
