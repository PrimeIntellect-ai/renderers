//! GPT-OSS (Harmony) renderer.
//!
//! Thin adapter over the `openai-harmony` Rust crate. Wire format is
//! harmony (channel-based, no BOS). The Python implementation goes
//! through the same library, so matching its conversion logic guarantees
//! byte-identical tokens.
//!
//! Architecture:
//!
//! - Holds a [`HarmonyEncoding`] (lazily loaded from
//!   [`HarmonyEncodingName::HarmonyGptOss`]) and a cache of the
//!   special-token ids it exposes.
//! - `render` builds a prefix conversation (SystemContent + DeveloperContent
//!   when a system message or tools are present) via
//!   `render_conversation`, then walks the remaining messages and renders
//!   each one individually via `render(msg)` so per-token attribution
//!   stays per-source-message.
//! - `parse_response` walks the completion tokens with our own scanner
//!   (token-id based) — matching what `renderers/parsing.py:parse_gpt_oss`
//!   does — so we don't need to manage a `StreamableParser`'s lifetime.
//!
//! This renderer does NOT need a HuggingFace `tokenizer.json`; the
//! harmony encoding embeds its own tiktoken-based tokenizer.

use std::sync::Arc;

use openai_harmony::chat::{
    Author, ChannelConfig, Conversation, DeveloperContent, Message as HarmonyMessage,
    ReasoningEffort, Role as HarmonyRole, SystemContent, ToolDescription,
};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};

use crate::bridge::{reject_assistant_in_extension, trim_to_turn_close};
use crate::thinking::should_preserve_past_thinking;
use crate::traits::Renderer;
use crate::types::{
    Message, ParsedResponse, ParsedToolCall, RenderError, RenderedTokens, SCAFFOLD_IDX,
    ToolArguments, ToolCallParseStatus, ToolSpec,
};

fn harmony_err<E: std::fmt::Display>(e: E) -> RenderError {
    RenderError::Invalid(format!("harmony: {e}"))
}

/// Builder for [`GptOssRenderer`].
#[derive(Debug, Clone)]
pub struct GptOssRendererBuilder {
    use_system_prompt: bool,
    reasoning_effort: ReasoningEffort,
    conversation_start_date: Option<String>,
    knowledge_cutoff: Option<String>,
    model_identity: Option<String>,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,
}

impl Default for GptOssRendererBuilder {
    fn default() -> Self {
        Self {
            use_system_prompt: true,
            reasoning_effort: ReasoningEffort::Medium,
            conversation_start_date: None,
            knowledge_cutoff: None,
            model_identity: None,
            preserve_all_thinking: false,
            preserve_thinking_between_tool_calls: false,
        }
    }
}

impl GptOssRendererBuilder {
    pub fn use_system_prompt(mut self, on: bool) -> Self {
        self.use_system_prompt = on;
        self
    }
    pub fn reasoning_effort(mut self, effort: &str) -> Result<Self, RenderError> {
        self.reasoning_effort = match effort.to_ascii_lowercase().as_str() {
            "low" => ReasoningEffort::Low,
            "medium" => ReasoningEffort::Medium,
            "high" => ReasoningEffort::High,
            other => {
                return Err(RenderError::Invalid(format!(
                    "unknown reasoning effort: {other}"
                )));
            }
        };
        Ok(self)
    }
    pub fn conversation_start_date(mut self, d: impl Into<String>) -> Self {
        self.conversation_start_date = Some(d.into());
        self
    }
    pub fn knowledge_cutoff(mut self, k: impl Into<String>) -> Self {
        self.knowledge_cutoff = Some(k.into());
        self
    }
    pub fn model_identity(mut self, m: impl Into<String>) -> Self {
        self.model_identity = Some(m.into());
        self
    }
    pub fn preserve_all_thinking(mut self, on: bool) -> Self {
        self.preserve_all_thinking = on;
        self
    }
    pub fn preserve_thinking_between_tool_calls(mut self, on: bool) -> Self {
        self.preserve_thinking_between_tool_calls = on;
        self
    }
    pub fn build(self) -> Result<GptOssRenderer, RenderError> {
        GptOssRenderer::new_with(self)
    }
}

#[derive(Debug, Clone)]
pub struct GptOssRenderer {
    enc: Arc<HarmonyEncoding>,
    use_system_prompt: bool,
    reasoning_effort: ReasoningEffort,
    conversation_start_date: String,
    knowledge_cutoff: Option<String>,
    model_identity: Option<String>,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,

    // Cached special-token ids — used by the parser and the generation prompt.
    start: u32,
    end: u32,
    return_tok: u32,
    call: u32,
    channel: u32,
    message: u32,
    #[allow(dead_code)]
    constrain: u32,

    stop_tokens: Vec<u32>,
}

impl GptOssRenderer {
    pub fn new() -> Result<Self, RenderError> {
        GptOssRendererBuilder::default().build()
    }
    pub fn builder() -> GptOssRendererBuilder {
        GptOssRendererBuilder::default()
    }

    fn new_with(cfg: GptOssRendererBuilder) -> Result<Self, RenderError> {
        let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).map_err(harmony_err)?;

        // Resolve special-token ids by encoding their canonical text and
        // asserting a single-token round-trip. The harmony encoding
        // exposes a `tokenizer()` accessor (tiktoken CoreBPE) so we use
        // its public special-token API. Bound to `enc` here directly so
        // the rest of the constructor doesn't need to name CoreBPE
        // (private outside the harmony crate).
        let resolve = |s: &str| -> Result<u32, RenderError> {
            let ids = enc.tokenizer().encode_with_special_tokens(s);
            if ids.len() != 1 {
                return Err(RenderError::MissingSpecialToken(s.to_string()));
            }
            // `Rank` in tiktoken is `u32`; no conversion needed.
            Ok(ids[0])
        };
        let start = resolve("<|start|>")?;
        let end = resolve("<|end|>")?;
        let return_tok = resolve("<|return|>")?;
        let call = resolve("<|call|>")?;
        let channel = resolve("<|channel|>")?;
        let message = resolve("<|message|>")?;
        let constrain = resolve("<|constrain|>")?;

        let start_date = cfg
            .conversation_start_date
            .clone()
            .unwrap_or_else(today_yyyy_mm_dd);

        Ok(Self {
            enc: Arc::new(enc),
            use_system_prompt: cfg.use_system_prompt,
            reasoning_effort: cfg.reasoning_effort,
            conversation_start_date: start_date,
            knowledge_cutoff: cfg.knowledge_cutoff,
            model_identity: cfg.model_identity,
            preserve_all_thinking: cfg.preserve_all_thinking,
            preserve_thinking_between_tool_calls: cfg.preserve_thinking_between_tool_calls,
            start,
            end,
            return_tok,
            call,
            channel,
            message,
            constrain,
            stop_tokens: vec![return_tok, call],
        })
    }

    /// Append rendered ids to `tokens`, attribute each to `msg_idx`.
    fn emit_render(
        &self,
        tokens: &mut Vec<u32>,
        indices: &mut Vec<i32>,
        msg_idx: i32,
        message: &HarmonyMessage,
    ) -> Result<(), RenderError> {
        let mut out: Vec<u32> = Vec::new();
        self.enc
            .render_into(message, &mut out, None)
            .map_err(harmony_err)?;
        let len = out.len();
        tokens.append(&mut out);
        indices.extend(std::iter::repeat_n(msg_idx, len));
        Ok(())
    }

    /// Encode a UTF-8 string via the harmony tokenizer, returning u32 ids.
    /// Helper so the call sites don't need to name CoreBPE (which is not
    /// re-exported from the harmony crate).
    fn encode_text(&self, text: &str) -> Vec<u32> {
        // `Rank` is `u32`; encode_with_special_tokens already returns Vec<u32>.
        self.enc.tokenizer().encode_with_special_tokens(text)
    }

    /// Decode a slice of token ids via the harmony tokenizer.
    fn decode_text(&self, ids: &[u32]) -> String {
        if ids.is_empty() {
            return String::new();
        }
        // `Rank` in tiktoken is `u32` — pass ids directly without casting.
        self.enc
            .tokenizer()
            .decode_utf8(ids.iter().copied())
            .unwrap_or_default()
    }

    fn render_conversation_tokens(
        &self,
        messages: Vec<HarmonyMessage>,
    ) -> Result<Vec<u32>, RenderError> {
        let convo = Conversation::from_messages(messages);
        let mut out: Vec<u32> = Vec::new();
        self.enc
            .render_conversation_into(convo.messages.iter(), &mut out, None)
            .map_err(harmony_err)?;
        Ok(out)
    }

    /// Build the harmony Author for tool messages — needs the function
    /// name, which we recover from `msg.name` (set client-side by
    /// `_attach_tool_call_names`).
    fn tool_author(msg: &Message) -> Author {
        let name = msg.name.as_deref().unwrap_or("unknown");
        let qualified: String = if name.starts_with("functions.") {
            name.to_string()
        } else {
            format!("functions.{name}")
        };
        Author {
            role: HarmonyRole::Tool,
            name: Some(qualified),
        }
    }

    fn message_to_harmony(&self, msg: &Message, preserve_thinking: bool) -> Vec<HarmonyMessage> {
        match msg.role.as_str() {
            "user" => vec![HarmonyMessage::from_role_and_content(
                HarmonyRole::User,
                msg.text_content().to_string(),
            )],
            "system" | "developer" => {
                let dev = DeveloperContent::new().with_instructions(msg.text_content());
                vec![HarmonyMessage::from_role_and_content(
                    HarmonyRole::Developer,
                    dev,
                )]
            }
            "tool" => {
                let m = HarmonyMessage::from_author_and_content(
                    Self::tool_author(msg),
                    msg.text_content().to_string(),
                )
                .with_recipient("assistant")
                .with_channel("commentary");
                vec![m]
            }
            "assistant" => self.assistant_to_harmony(msg, preserve_thinking),
            _ => {
                let dev = DeveloperContent::new().with_instructions(msg.text_content());
                vec![HarmonyMessage::from_role_and_content(
                    HarmonyRole::Developer,
                    dev,
                )]
            }
        }
    }

    fn assistant_to_harmony(&self, msg: &Message, preserve_thinking: bool) -> Vec<HarmonyMessage> {
        let mut out: Vec<HarmonyMessage> = Vec::new();

        if preserve_thinking {
            if let Some(reasoning) = msg.reasoning_content.as_deref() {
                if !reasoning.is_empty() {
                    let m = HarmonyMessage::from_role_and_content(
                        HarmonyRole::Assistant,
                        reasoning.to_string(),
                    )
                    .with_channel("analysis");
                    out.push(m);
                }
            }
        }

        // Text content goes on the `final` channel.
        let text = msg.text_content();
        if !text.is_empty() {
            let m = HarmonyMessage::from_role_and_content(HarmonyRole::Assistant, text.to_string())
                .with_channel("final");
            out.push(m);
        }

        // Each tool_call becomes its own assistant message on the
        // commentary channel with recipient=functions.<name>.
        for tc in &msg.tool_calls {
            let name = &tc.function.name;
            let args = match &tc.function.arguments {
                ToolArguments::Raw(s) => s.clone(),
                ToolArguments::Object(v) => serde_json::to_string(v).unwrap_or_default(),
            };
            let recipient = if name.starts_with("functions.") {
                name.clone()
            } else {
                format!("functions.{name}")
            };
            let m = HarmonyMessage::from_role_and_content(HarmonyRole::Assistant, args)
                .with_channel("commentary")
                .with_recipient(recipient);
            out.push(m);
        }

        // Empty assistant with no text and no tool_calls: emit empty
        // final-channel message so per-token attribution still produces
        // at least one token slot.
        if out.is_empty() {
            let m = HarmonyMessage::from_role_and_content(HarmonyRole::Assistant, String::new())
                .with_channel("final");
            out.push(m);
        }

        out
    }

    fn tool_to_description(tool: &ToolSpec) -> ToolDescription {
        ToolDescription::new(
            tool.name.as_str(),
            tool.description.as_str(),
            Some(tool.parameters.clone()),
        )
    }

    fn build_system_content(&self) -> SystemContent {
        let mut s = SystemContent::new().with_reasoning_effort(self.reasoning_effort);
        s = s.with_conversation_start_date(self.conversation_start_date.as_str());
        if let Some(k) = &self.knowledge_cutoff {
            s = s.with_knowledge_cutoff(k.as_str());
        }
        if let Some(m) = &self.model_identity {
            s = s.with_model_identity(m.as_str());
        }
        s
    }

    fn emit_generation_prompt(&self, tokens: &mut Vec<u32>, indices: &mut Vec<i32>) {
        tokens.push(self.start);
        indices.push(SCAFFOLD_IDX);
        // "assistant" + <|channel|> + "analysis" + <|message|>
        for id in self.encode_text("assistant") {
            tokens.push(id);
            indices.push(SCAFFOLD_IDX);
        }
        tokens.push(self.channel);
        indices.push(SCAFFOLD_IDX);
        for id in self.encode_text("analysis") {
            tokens.push(id);
            indices.push(SCAFFOLD_IDX);
        }
        tokens.push(self.message);
        indices.push(SCAFFOLD_IDX);
    }
}

impl Renderer for GptOssRenderer {
    fn render(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError> {
        if messages.is_empty() {
            return Err(RenderError::EmptyMessages);
        }
        let mut tokens: Vec<u32> = Vec::with_capacity(messages.len() * 256);
        let mut indices: Vec<i32> = Vec::with_capacity(messages.len() * 256);

        let first_system_idx = messages.iter().position(|m| m.role == "system");

        // Prefix: SystemContent + DeveloperContent (when tools or a
        // caller-supplied system are present).
        let mut prefix_msgs: Vec<HarmonyMessage> = Vec::new();
        if self.use_system_prompt {
            let sys = self.build_system_content();
            let sys = match tools {
                Some(t) if !t.is_empty() => {
                    sys.with_channel_config(ChannelConfig::require_channels([
                        "analysis",
                        "commentary",
                        "final",
                    ]))
                }
                _ => sys,
            };
            prefix_msgs.push(HarmonyMessage::from_role_and_content(
                HarmonyRole::System,
                sys,
            ));
        }
        let has_dev = first_system_idx.is_some() || tools.map(|t| !t.is_empty()).unwrap_or(false);
        if has_dev {
            let mut dev = DeveloperContent::new();
            if let Some(idx) = first_system_idx {
                let instr = messages[idx].text_content();
                if !instr.is_empty() {
                    dev = dev.with_instructions(instr);
                }
            }
            if let Some(t) = tools {
                if !t.is_empty() {
                    let descs: Vec<ToolDescription> =
                        t.iter().map(Self::tool_to_description).collect();
                    dev = dev.with_function_tools(descs);
                }
            }
            prefix_msgs.push(HarmonyMessage::from_role_and_content(
                HarmonyRole::Developer,
                dev,
            ));
        }
        if !prefix_msgs.is_empty() {
            let prefix_tokens = self.render_conversation_tokens(prefix_msgs)?;
            let attr_idx: i32 = first_system_idx.map(|i| i as i32).unwrap_or(SCAFFOLD_IDX);
            for id in prefix_tokens {
                tokens.push(id);
                indices.push(attr_idx);
            }
        }

        // Body
        let last_idx = messages.len() - 1;
        for (i, msg) in messages.iter().enumerate() {
            if Some(i) == first_system_idx {
                continue;
            }
            let preserve_thinking = msg.role == "assistant"
                && should_preserve_past_thinking(
                    messages,
                    i,
                    self.preserve_all_thinking,
                    self.preserve_thinking_between_tool_calls,
                );
            for hm in self.message_to_harmony(msg, preserve_thinking) {
                self.emit_render(&mut tokens, &mut indices, i as i32, &hm)?;
            }
        }

        // Terminal close: if the conversation ends on a plain assistant
        // turn (no tool_calls) and we're not asking for a generation
        // prompt, swap the trailing <|end|> for <|return|> — matches
        // apply_chat_template.
        if !add_generation_prompt
            && last_idx < messages.len()
            && messages[last_idx].role == "assistant"
            && messages[last_idx].tool_calls.is_empty()
            && tokens.last().copied() == Some(self.end)
        {
            *tokens.last_mut().expect("non-empty") = self.return_tok;
        }

        if add_generation_prompt {
            self.emit_generation_prompt(&mut tokens, &mut indices);
        }

        Ok(RenderedTokens {
            token_ids: tokens,
            message_indices: indices,
            multi_modal_data: None,
        })
    }

    fn parse_response(&self, token_ids: &[u32]) -> ParsedResponse {
        // Walk tokens block-by-block: `<|start|>{header}<|message|>{body}{terminator}`.
        // Terminator is one of `<|start|>` (next block), `<|end|>`, `<|call|>`.
        // `<|return|>` truncates the entire response.
        let return_pos = token_ids.iter().position(|&t| t == self.return_tok);
        let ids: &[u32] = match return_pos {
            Some(p) => &token_ids[..p],
            None => token_ids,
        };

        let mut reasoning_parts: Vec<String> = Vec::new();
        let mut content_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ParsedToolCall> = Vec::new();

        let mut i = 0usize;
        while i < ids.len() {
            if ids[i] != self.start {
                i += 1;
                continue;
            }
            let block_start = i;
            let Some(msg_pos) = ids[i + 1..]
                .iter()
                .position(|&t| t == self.message)
                .map(|p| p + i + 1)
            else {
                break;
            };
            let header_ids = &ids[i + 1..msg_pos];
            let header_text = self.decode_text(header_ids);

            let body_start = msg_pos + 1;
            let body_end = ids[body_start..]
                .iter()
                .position(|&t| t == self.start || t == self.end || t == self.call)
                .map(|p| p + body_start)
                .unwrap_or(ids.len());
            let body_closed =
                body_end < ids.len() && (ids[body_end] == self.end || ids[body_end] == self.call);
            let body_text = self.decode_text(&ids[body_start..body_end]);

            // Channel: look for <|channel|>NAME in header — NAME is the
            // text between the channel token and the next whitespace /
            // special token.
            let channel = header_ids
                .iter()
                .position(|&t| t == self.channel)
                .map(|p| {
                    let after = &header_ids[p + 1..];
                    // Take tokens until newline/space — but since header
                    // is short, just decode the rest and split.
                    self.decode_text(after).trim().to_string()
                })
                .unwrap_or_default();

            // Recipient: header text may contain "to=functions.NAME"
            let recipient: Option<&str> = header_text.split("to=").nth(1).map(|s| {
                s.split(|c: char| c.is_whitespace() || c == '<')
                    .next()
                    .unwrap_or("")
            });

            if let Some(r) = recipient {
                if let Some(tool_name) = r.strip_prefix("functions.") {
                    let block_end = if body_closed { body_end + 1 } else { body_end };
                    let span = block_start..block_end;
                    match serde_json::from_str::<serde_json::Value>(&body_text) {
                        Ok(v) => {
                            tool_calls.push(ParsedToolCall {
                                raw: body_text.clone(),
                                name: Some(tool_name.to_string()),
                                arguments: Some(ToolArguments::Object(v)),
                                token_span: Some(span),
                                status: ToolCallParseStatus::Ok,
                                ..Default::default()
                            });
                        }
                        Err(_) => {
                            tool_calls.push(ParsedToolCall {
                                raw: body_text.clone(),
                                name: Some(tool_name.to_string()),
                                arguments: Some(ToolArguments::Raw(body_text.clone())),
                                token_span: Some(span),
                                status: ToolCallParseStatus::InvalidJson,
                                ..Default::default()
                            });
                        }
                    }
                    i = if body_closed { body_end + 1 } else { body_end };
                    continue;
                }
            }

            // analysis → reasoning_content; everything else (final,
            // commentary without a tool recipient, missing channel)
            // collapses into the visible content stream.
            match channel.split_whitespace().next() {
                Some("analysis") => reasoning_parts.push(body_text),
                _ => content_parts.push(body_text),
            }

            i = if body_closed { body_end + 1 } else { body_end };
        }

        let reasoning_content = if reasoning_parts.is_empty() {
            None
        } else {
            Some(reasoning_parts.join("").trim().to_string()).filter(|s| !s.is_empty())
        };

        ParsedResponse {
            content: content_parts.join("").trim().to_string(),
            reasoning_content,
            tool_calls,
        }
    }

    fn stop_token_ids(&self) -> &[u32] {
        &self.stop_tokens
    }

    fn bridge_to_next_turn(
        &self,
        previous_prompt_ids: &[u32],
        previous_completion_ids: &[u32],
        new_messages: &[Message],
        _tools: Option<&[ToolSpec]>,
    ) -> Result<Option<RenderedTokens>, RenderError> {
        if previous_prompt_ids.is_empty()
            || new_messages.is_empty()
            || reject_assistant_in_extension(new_messages)
        {
            return Ok(None);
        }
        let Some(previous_ids) = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            &[self.return_tok, self.call],
            Some(self.end),
        ) else {
            return Ok(None);
        };

        let mut ext: Vec<u32> = Vec::new();
        for msg in new_messages {
            match msg.role.as_str() {
                "tool" | "user" | "system" | "developer" => {}
                _ => return Ok(None),
            }
            for hm in self.message_to_harmony(msg, false) {
                let mut out: Vec<u32> = Vec::new();
                self.enc
                    .render_into(&hm, &mut out, None)
                    .map_err(harmony_err)?;
                ext.extend(out);
            }
        }

        // Generation prompt
        ext.push(self.start);
        ext.extend(self.encode_text("assistant"));
        ext.push(self.channel);
        ext.extend(self.encode_text("analysis"));
        ext.push(self.message);

        let mut out = Vec::with_capacity(previous_ids.len() + ext.len());
        out.extend_from_slice(&previous_ids);
        out.extend_from_slice(&ext);
        Ok(Some(RenderedTokens {
            token_ids: out,
            message_indices: Vec::new(),
            multi_modal_data: None,
        }))
    }
}

fn today_yyyy_mm_dd() -> String {
    // Avoid pulling chrono — use std::time::SystemTime and a small
    // conversion that's good enough for "today" in UTC.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    // 1970-01-01 + days
    let (y, m, d) = civil_from_days(days as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Convert days since 1970-01-01 to (year, month, day) — Howard Hinnant's
/// algorithm, public-domain.
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i32 + era as i32 * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
