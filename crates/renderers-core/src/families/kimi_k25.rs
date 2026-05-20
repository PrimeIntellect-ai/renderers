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
use crate::traits::Renderer;
use crate::types::{
    Message, ParsedResponse, RenderError, RenderedTokens, ToolArguments, ToolSpec,
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
        let tool_calls_section_end =
            tokenizer.token_to_id_strict("<|tool_calls_section_end|>")?;
        let tool_call_begin = tokenizer.token_to_id_strict("<|tool_call_begin|>")?;
        let tool_call_argument_begin =
            tokenizer.token_to_id_strict("<|tool_call_argument_begin|>")?;
        let tool_call_end = tokenizer.token_to_id_strict("<|tool_call_end|>")?;

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
            stop_tokens: vec![im_end],
        })
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

    fn emit_tool_body(&self, buf: &mut RenderBuf<'_>, msg: &Message, msg_idx: i32) -> Result<(), RenderError> {
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
                "tool" => self.emit_tool_body(&mut buf, msg, idx)?,
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
                "tool" => self.emit_tool_body(&mut buf, msg, idx)?,
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
}
