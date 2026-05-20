//! Kimi K2.5 renderer (text-only path, no tools).
//!
//! Port of `renderers/kimi_k25.py` covering the most common call shape:
//! chat without function-calling tools and without images. The path with
//! TypeScript-style tool declarations and the multimodal path are
//! deferred to Phase 5 (the Python shim keeps those on the pure-Python
//! implementation for now).
//!
//! Distinctive features vs Kimi K2:
//!
//! - Generation prompt prefills `<think>` (enable_thinking=True) or the
//!   empty block `<think></think>` (enable_thinking=False) to control
//!   thinking mode at sample time. `<think>` and `</think>` may be
//!   multi-token; the renderer encodes them as text.
//! - Assistant body uses the hist/suffix split: the last non-tool-call
//!   assistant + all later assistants keep `reasoning_content`;
//!   historical assistants collapse to a literal `<think></think>`.
//! - Default system message is the same as K2
//!   ("You are Kimi, an AI assistant created by Moonshot AI.") but the
//!   Python class doesn't auto-inject it — neither does this port.

use crate::bridge::{reject_assistant_in_extension, trim_to_turn_close};
use crate::emit::RenderBuf;
use crate::parsing::kimi_k2::parse_kimi_k2;
use crate::thinking::should_preserve_past_thinking;
use crate::tokenizer::Tokenizer;
use crate::traits::{MultimodalRenderer, Renderer};
use crate::types::{
    MediaBundle, MediaItem, Message, Modality, MultiModalData, ParsedResponse, PlaceholderRange,
    RenderError, RenderedTokens, ToolArguments, ToolSpec,
};

#[derive(Debug, Clone)]
pub struct KimiK25RendererBuilder {
    enable_thinking: bool,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,
}

impl Default for KimiK25RendererBuilder {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            preserve_all_thinking: false,
            preserve_thinking_between_tool_calls: false,
        }
    }
}

impl KimiK25RendererBuilder {
    pub fn enable_thinking(mut self, on: bool) -> Self {
        self.enable_thinking = on;
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
    pub fn build(self, tokenizer: Tokenizer) -> Result<KimiK25Renderer, RenderError> {
        KimiK25Renderer::new_with(tokenizer, self)
    }
}

#[derive(Debug, Clone)]
pub struct KimiK25Renderer {
    tokenizer: Tokenizer,
    enable_thinking: bool,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,

    im_user: u32,
    im_assistant: u32,
    im_system: u32,
    im_middle: u32,
    im_end: u32,
    tool_calls_section_begin: u32,
    tool_calls_section_end: u32,
    tool_call_begin: u32,
    tool_call_argument_begin: u32,
    tool_call_end: u32,

    // Media tokens — present on K2.5 tokenizers, absent on K2 proper.
    // When absent, as_multimodal() returns None.
    media_begin: Option<u32>,
    media_content: Option<u32>,
    media_pad: Option<u32>,
    media_end: Option<u32>,
    mm_token_type_ids: Vec<(u32, u8)>,

    stop_tokens: Vec<u32>,
}

impl KimiK25Renderer {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, RenderError> {
        KimiK25RendererBuilder::default().build(tokenizer)
    }
    pub fn builder() -> KimiK25RendererBuilder {
        KimiK25RendererBuilder::default()
    }

    fn new_with(tokenizer: Tokenizer, cfg: KimiK25RendererBuilder) -> Result<Self, RenderError> {
        let im_user = tokenizer.token_to_id_strict("<|im_user|>")?;
        let im_assistant = tokenizer.token_to_id_strict("<|im_assistant|>")?;
        let im_system = tokenizer.token_to_id_strict("<|im_system|>")?;
        let im_middle = tokenizer.token_to_id_strict("<|im_middle|>")?;
        let im_end = tokenizer.token_to_id_strict("<|im_end|>")?;
        let tool_calls_section_begin =
            tokenizer.token_to_id_strict("<|tool_calls_section_begin|>")?;
        let tool_calls_section_end = tokenizer.token_to_id_strict("<|tool_calls_section_end|>")?;
        let tool_call_begin = tokenizer.token_to_id_strict("<|tool_call_begin|>")?;
        let tool_call_argument_begin =
            tokenizer.token_to_id_strict("<|tool_call_argument_begin|>")?;
        let tool_call_end = tokenizer.token_to_id_strict("<|tool_call_end|>")?;

        // Media tokens optional — K2 proper doesn't ship them.
        let media_begin = tokenizer.token_to_id("<|media_begin|>");
        let media_content = tokenizer.token_to_id("<|media_content|>");
        let media_pad = tokenizer.token_to_id("<|media_pad|>");
        let media_end = tokenizer.token_to_id("<|media_end|>");
        let mut mm_token_type_ids: Vec<(u32, u8)> = Vec::new();
        if let Some(p) = media_pad {
            mm_token_type_ids.push((p, 1)); // image marker; K2.5 handles video via the same pad
        }

        Ok(Self {
            tokenizer,
            enable_thinking: cfg.enable_thinking,
            preserve_all_thinking: cfg.preserve_all_thinking,
            preserve_thinking_between_tool_calls: cfg.preserve_thinking_between_tool_calls,
            im_user,
            im_assistant,
            im_system,
            im_middle,
            im_end,
            tool_calls_section_begin,
            tool_calls_section_end,
            tool_call_begin,
            tool_call_argument_begin,
            tool_call_end,
            media_begin,
            media_content,
            media_pad,
            media_end,
            mm_token_type_ids,
            stop_tokens: vec![im_end],
        })
    }

    /// True when the loaded tokenizer ships the K2.5 media tokens.
    pub fn supports_multimodal(&self) -> bool {
        self.media_begin.is_some()
            && self.media_content.is_some()
            && self.media_pad.is_some()
            && self.media_end.is_some()
    }

    fn args_to_string(args: &ToolArguments) -> String {
        match args {
            ToolArguments::Raw(s) => s.clone(),
            ToolArguments::Object(v) => serde_json::to_string(v).unwrap_or_else(|_| "{}".into()),
        }
    }

    fn role_token(&self, role: &str) -> u32 {
        match role {
            "user" => self.im_user,
            "assistant" => self.im_assistant,
            _ => self.im_system,
        }
    }

    /// Extract `(reasoning_content, text_content)` from a message,
    /// honouring the explicit `reasoning_content` field and the inline
    /// `<think>...</think>` tag fallback. Mirrors the Python K2.5
    /// `_render_assistant_body` extraction.
    fn extract_reasoning(msg: &Message) -> (String, String) {
        if let Some(r) = &msg.reasoning_content {
            return (r.clone(), msg.text_content().to_string());
        }
        let content = msg.text_content();
        if let Some((before, after)) = content.split_once("</think>") {
            let reasoning = if let Some((_, inner)) = before.rsplit_once("<think>") {
                inner.to_string()
            } else {
                before.to_string()
            };
            return (reasoning, after.trim_start_matches('\n').to_string());
        }
        (String::new(), content.to_string())
    }

    fn emit_assistant_body(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        msg_idx: i32,
        is_suffix: bool,
        preserve_thinking: bool,
    ) -> Result<(), RenderError> {
        let (reasoning_content, text_content) = Self::extract_reasoning(msg);

        // hist/suffix split: hist drops reasoning, suffix preserves it.
        if is_suffix || (preserve_thinking && !reasoning_content.is_empty()) {
            let mut s = String::with_capacity(reasoning_content.len() + 16);
            s.push_str("<think>");
            s.push_str(&reasoning_content);
            s.push_str("</think>");
            buf.text(&s, msg_idx)?;
        } else {
            buf.text("<think></think>", msg_idx)?;
        }
        buf.text(&text_content, msg_idx)?;

        if !msg.tool_calls.is_empty() {
            buf.special(self.tool_calls_section_begin, msg_idx);
            for tc in &msg.tool_calls {
                let args_str = Self::args_to_string(&tc.function.arguments);
                let tool_id = tc.id.clone().unwrap_or_default();
                buf.special(self.tool_call_begin, msg_idx);
                buf.text(&tool_id, msg_idx)?;
                buf.special(self.tool_call_argument_begin, msg_idx);
                buf.text(&args_str, msg_idx)?;
                buf.special(self.tool_call_end, msg_idx);
            }
            buf.special(self.tool_calls_section_end, msg_idx);
        }
        Ok(())
    }

    fn emit_tool_body(
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        msg_idx: i32,
    ) -> Result<(), RenderError> {
        let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");
        let mut header = String::with_capacity(tool_call_id.len() + 16);
        header.push_str("## Return of ");
        header.push_str(tool_call_id);
        header.push('\n');
        buf.text(&header, msg_idx)?;
        let content = msg.text_content();
        if !content.is_empty() {
            buf.text(content, msg_idx)?;
        }
        Ok(())
    }
}

impl Renderer for KimiK25Renderer {
    fn render(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError> {
        if messages.is_empty() {
            return Err(RenderError::EmptyMessages);
        }
        // Tools route to Python — the TS-style declaration formatter
        // (~270 lines) isn't ported yet. The Python shim avoids native
        // routing when tools are present, so this is a hard error if we
        // got here with tools.
        if tools.map(|t| !t.is_empty()).unwrap_or(false) {
            return Err(RenderError::Invalid(
                "Kimi K2.5 with tools not supported on the native path yet; the Python shim should route to pure Python in this case".into(),
            ));
        }

        let mut buf = RenderBuf::new(&self.tokenizer, messages.len().max(1) * 256);

        // Find last non-tool-call assistant for the hist/suffix split
        let mut last_non_tc_assistant: i32 = -1;
        for (i, m) in messages.iter().enumerate().rev() {
            if m.role == "assistant" && m.tool_calls.is_empty() {
                last_non_tc_assistant = i as i32;
                break;
            }
        }

        for (i, msg) in messages.iter().enumerate() {
            let idx = i as i32;
            buf.special(self.role_token(&msg.role), idx);
            // K2.5 uses `msg.name or role` as the role-name literal
            let role_name = msg.name.as_deref().unwrap_or(&msg.role);
            buf.text(role_name, idx)?;
            buf.special(self.im_middle, idx);

            match msg.role.as_str() {
                "assistant" => {
                    let is_suffix = idx > last_non_tc_assistant;
                    let preserve_thinking = should_preserve_past_thinking(
                        messages,
                        i,
                        self.preserve_all_thinking,
                        self.preserve_thinking_between_tool_calls,
                    );
                    self.emit_assistant_body(&mut buf, msg, idx, is_suffix, preserve_thinking)?;
                }
                "tool" => Self::emit_tool_body(&mut buf, msg, idx)?,
                _ => {
                    let content = msg.text_content();
                    if !content.is_empty() {
                        buf.text(content, idx)?;
                    }
                }
            }
            buf.special(self.im_end, idx);
        }

        // Generation prompt
        if add_generation_prompt {
            buf.scaffold_special(self.im_assistant);
            buf.scaffold_text("assistant")?;
            buf.scaffold_special(self.im_middle);
            if self.enable_thinking {
                buf.scaffold_text("<think>")?;
            } else {
                buf.scaffold_text("<think></think>")?;
            }
        }

        Ok(buf.into_rendered())
    }

    fn parse_response(&self, token_ids: &[u32]) -> ParsedResponse {
        // K2.5 reuses the K2 parser shape; only differences are the
        // thinking-tag handling, which the K2 parser already does via the
        // decoded-text branch.
        parse_kimi_k2(
            &self.tokenizer,
            token_ids,
            &self.stop_tokens,
            self.tool_calls_section_begin,
            self.tool_calls_section_end,
            self.tool_call_begin,
            self.tool_call_argument_begin,
            self.tool_call_end,
        )
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
            &self.stop_tokens,
            Some(self.im_end),
        ) else {
            return Ok(None);
        };

        let mut buf = RenderBuf::new(&self.tokenizer, new_messages.len().max(1) * 256);
        for (i, msg) in new_messages.iter().enumerate() {
            let idx = i as i32;
            buf.special(self.role_token(&msg.role), idx);
            let role_name = msg.name.as_deref().unwrap_or(&msg.role);
            buf.text(role_name, idx)?;
            buf.special(self.im_middle, idx);
            match msg.role.as_str() {
                "user" | "system" => {
                    let content = msg.text_content();
                    if !content.is_empty() {
                        buf.text(content, idx)?;
                    }
                }
                "tool" => Self::emit_tool_body(&mut buf, msg, idx)?,
                _ => return Ok(None),
            }
            buf.special(self.im_end, idx);
        }

        // Generation prompt
        buf.scaffold_special(self.im_assistant);
        buf.scaffold_text("assistant")?;
        buf.scaffold_special(self.im_middle);
        if self.enable_thinking {
            buf.scaffold_text("<think>")?;
        } else {
            buf.scaffold_text("<think></think>")?;
        }

        let ext = buf.into_token_ids();
        let mut out = Vec::with_capacity(previous_ids.len() + ext.len());
        out.extend_from_slice(&previous_ids);
        out.extend_from_slice(&ext);
        Ok(Some(RenderedTokens {
            token_ids: out,
            message_indices: Vec::new(),
            multi_modal_data: None,
        }))
    }

    fn as_multimodal(&self) -> Option<&dyn MultimodalRenderer> {
        if self.supports_multimodal() {
            Some(self)
        } else {
            None
        }
    }
}

// ── Multimodal implementation ─────────────────────────────────────────
//
// Kimi K2.5's placeholder shape diverges from Qwen-VL: each image gets
// exactly ONE `<|media_pad|>` token in the input stream, regardless of
// image size. The model's vision encoder expands per-patch attention
// internally from `pixel_values` + `grid_thws`. The renderer's job is
// just to emit the per-image wrapper:
//
//     <|media_begin|>image<|media_content|><|media_pad|><|media_end|>\n
//
// and accumulate the corresponding placeholder ranges + opaque payloads.

impl KimiK25Renderer {
    fn emit_media_item(
        &self,
        buf: &mut RenderBuf<'_>,
        idx: i32,
        item: &MediaItem,
        mm: &mut MultiModalData,
    ) -> Result<(), RenderError> {
        let begin = self
            .media_begin
            .ok_or_else(|| RenderError::MissingSpecialToken("<|media_begin|>".into()))?;
        let content = self
            .media_content
            .ok_or_else(|| RenderError::MissingSpecialToken("<|media_content|>".into()))?;
        let pad = self
            .media_pad
            .ok_or_else(|| RenderError::MissingSpecialToken("<|media_pad|>".into()))?;
        let end = self
            .media_end
            .ok_or_else(|| RenderError::MissingSpecialToken("<|media_end|>".into()))?;

        let label = match item.modality {
            Modality::Image => "image",
            Modality::Video => "video",
        };

        buf.special(begin, idx);
        buf.text(label, idx)?;
        buf.special(content, idx);
        let offset = buf.len();
        buf.special(pad, idx);
        buf.special(end, idx);
        buf.text("\n", idx)?;

        // Always exactly 1 placeholder in the stream, regardless of
        // image size — that's the K2.5 convention.
        let key = item.modality.as_str().to_string();
        mm.mm_hashes
            .entry(key.clone())
            .or_default()
            .push(item.hash.clone());
        mm.mm_placeholders
            .entry(key.clone())
            .or_default()
            .push(PlaceholderRange { offset, length: 1 });
        mm.mm_items
            .entry(key)
            .or_default()
            .push(item.hf_payload.clone());
        Ok(())
    }

    fn emit_user_body_with_media<'m>(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        msg_idx: i32,
        media_iter: &mut impl Iterator<Item = &'m MediaItem>,
        mm: &mut MultiModalData,
    ) -> Result<(), RenderError> {
        match &msg.content {
            crate::types::Content::Null => {
                for item in media_iter.by_ref() {
                    self.emit_media_item(buf, msg_idx, item, mm)?;
                }
            }
            crate::types::Content::Text(s) => {
                // Plain-text + attached media: emit images first, then
                // text. Same convention as Qwen-VL when the caller
                // doesn't pass a structured content list.
                for item in media_iter.by_ref() {
                    self.emit_media_item(buf, msg_idx, item, mm)?;
                }
                if !s.is_empty() {
                    buf.text(s, msg_idx)?;
                }
            }
            crate::types::Content::Parts(parts) => {
                use crate::types::ContentPart;
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            if !text.is_empty() {
                                buf.text(text, msg_idx)?;
                            }
                        }
                        ContentPart::Thinking { .. } => {}
                        ContentPart::Image(_) | ContentPart::Video(_) => {
                            let item = media_iter.next().ok_or_else(|| {
                                RenderError::Invalid(
                                    "K2.5 message content lists more media parts than the MediaBundle provides".into(),
                                )
                            })?;
                            self.emit_media_item(buf, msg_idx, item, mm)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl MultimodalRenderer for KimiK25Renderer {
    fn mm_token_type_id_map(&self) -> &[(u32, u8)] {
        &self.mm_token_type_ids
    }

    fn render_with_media(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        media: &MediaBundle,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError> {
        if media.is_empty() {
            return self.render(messages, tools, add_generation_prompt);
        }
        if messages.is_empty() {
            return Err(RenderError::EmptyMessages);
        }
        if tools.map(|t| !t.is_empty()).unwrap_or(false) {
            return Err(RenderError::Invalid(
                "Kimi K2.5 with tools not supported on the native path yet".into(),
            ));
        }

        // Per-message media iterator. The bundle is flat (message_idx,
        // item), and K2.5 doesn't auto-inject system messages, so the
        // indices align directly with the caller's input.
        let mut buf = RenderBuf::new(&self.tokenizer, messages.len().max(1) * 256);
        let mut mm = MultiModalData::default();

        let mut last_non_tc_assistant: i32 = -1;
        for (i, m) in messages.iter().enumerate().rev() {
            if m.role == "assistant" && m.tool_calls.is_empty() {
                last_non_tc_assistant = i as i32;
                break;
            }
        }

        for (i, msg) in messages.iter().enumerate() {
            let idx = i as i32;
            buf.special(self.role_token(&msg.role), idx);
            let role_name = msg.name.as_deref().unwrap_or(&msg.role);
            buf.text(role_name, idx)?;
            buf.special(self.im_middle, idx);

            match msg.role.as_str() {
                "assistant" => {
                    let is_suffix = idx > last_non_tc_assistant;
                    let preserve_thinking = should_preserve_past_thinking(
                        messages,
                        i,
                        self.preserve_all_thinking,
                        self.preserve_thinking_between_tool_calls,
                    );
                    self.emit_assistant_body(&mut buf, msg, idx, is_suffix, preserve_thinking)?;
                }
                "tool" => Self::emit_tool_body(&mut buf, msg, idx)?,
                _ => {
                    // user / system / other — interleave media inline
                    let mut media_iter = media
                        .items
                        .iter()
                        .filter_map(|(m, it)| (*m == i).then_some(it));
                    self.emit_user_body_with_media(&mut buf, msg, idx, &mut media_iter, &mut mm)?;
                    if media_iter.next().is_some() {
                        return Err(RenderError::Invalid(format!(
                            "MediaBundle has more items for message {i} than the content's media parts"
                        )));
                    }
                }
            }
            buf.special(self.im_end, idx);
        }

        if add_generation_prompt {
            buf.scaffold_special(self.im_assistant);
            buf.scaffold_text("assistant")?;
            buf.scaffold_special(self.im_middle);
            if self.enable_thinking {
                buf.scaffold_text("<think>")?;
            } else {
                buf.scaffold_text("<think></think>")?;
            }
        }

        let mut out = buf.into_rendered();
        if !mm.is_empty() {
            out.multi_modal_data = Some(mm);
        }
        Ok(out)
    }

    fn bridge_to_next_turn_with_media(
        &self,
        previous_prompt_ids: &[u32],
        previous_completion_ids: &[u32],
        new_messages: &[Message],
        tools: Option<&[ToolSpec]>,
        new_media: &MediaBundle,
        _previous_multi_modal_data: Option<&MultiModalData>,
    ) -> Result<Option<RenderedTokens>, RenderError> {
        if !new_media.is_empty() {
            // Same Phase 5a caveat as Qwen3.5: bridging media-bearing
            // new turns is unsafe under truncation. Fall back to a full
            // re-render.
            return Ok(None);
        }
        self.bridge_to_next_turn(
            previous_prompt_ids,
            previous_completion_ids,
            new_messages,
            tools,
        )
    }
}
