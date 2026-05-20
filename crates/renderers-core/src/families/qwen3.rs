//! Qwen3 renderer. Port of `renderers/qwen3.py`.
//!
//! Byte-for-byte identical output to the Python version — the
//! `test_render_ids` / `test_bridge` / `test_roundtrip` golden suites are
//! the contract.
//!
//! # Performance notes
//!
//! - Special-token ids are resolved once at construction and cached on
//!   the struct. Zero per-call lookup cost.
//! - The render buffer is sized to `messages.len() * 256` up front; this
//!   covers ~99% of multi-turn conversations with no realloc.
//! - The tools header / footer are static `&str` constants — no
//!   per-call allocation.
//! - Tool-call argument serialisation goes through `serde_json` directly,
//!   ~5–10× faster than Python's `json.dumps` for the JSON sizes typical
//!   here.

use crate::bridge::{reject_assistant_in_extension, trim_to_turn_close};
use crate::emit::RenderBuf;
use crate::json::{to_string_python, tool_spec_template_value};
use crate::parsing::qwen3::parse_qwen3;
use crate::thinking::should_preserve_past_thinking;
use crate::tokenizer::Tokenizer;
use crate::traits::Renderer;
use crate::types::{
    Message, ParsedResponse, RenderError, RenderedTokens, SCAFFOLD_IDX, ToolArguments, ToolSpec,
};

const TOOLS_HEADER: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";

const TOOLS_FOOTER: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

const GEN_PROMPT_NO_THINKING_SUFFIX: &str = "<think>\n\n</think>\n\n";

/// Builder for [`Qwen3Renderer`]. Use this to surface the rare optional
/// flags without polluting the most common constructor.
#[derive(Debug, Clone)]
pub struct Qwen3RendererBuilder {
    enable_thinking: bool,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,
}

impl Default for Qwen3RendererBuilder {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            preserve_all_thinking: false,
            preserve_thinking_between_tool_calls: false,
        }
    }
}

impl Qwen3RendererBuilder {
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

    pub fn build(self, tokenizer: Tokenizer) -> Result<Qwen3Renderer, RenderError> {
        Qwen3Renderer::new_with(tokenizer, &self)
    }
}

/// Deterministic Qwen3 renderer.
#[derive(Debug, Clone)]
pub struct Qwen3Renderer {
    tokenizer: Tokenizer,
    enable_thinking: bool,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,

    im_start: u32,
    im_end: u32,
    /// Cached for parity with the Python `_endoftext` field; the
    /// stop-token set already encodes the same id, so this is unused
    /// directly but kept for debug parity.
    #[allow(dead_code)]
    endoftext: u32,
    tool_call: u32,
    tool_call_end: u32,
    tool_response: u32,
    tool_response_end: u32,

    /// Cached stop tokens (`im_end`, `endoftext`) for `stop_token_ids`
    /// and bridge close-token sets. Two-element vector held by-value
    /// per renderer instance.
    stop_tokens: Vec<u32>,
}

impl Qwen3Renderer {
    /// Convenience constructor with all defaults.
    pub fn new(tokenizer: Tokenizer) -> Result<Self, RenderError> {
        Qwen3RendererBuilder::default().build(tokenizer)
    }

    pub fn builder() -> Qwen3RendererBuilder {
        Qwen3RendererBuilder::default()
    }

    fn new_with(tokenizer: Tokenizer, cfg: &Qwen3RendererBuilder) -> Result<Self, RenderError> {
        let im_start = tokenizer.token_to_id_strict("<|im_start|>")?;
        let im_end = tokenizer.token_to_id_strict("<|im_end|>")?;
        let endoftext = tokenizer.token_to_id_strict("<|endoftext|>")?;
        let tool_call = tokenizer.token_to_id_strict("<tool_call>")?;
        let tool_call_end = tokenizer.token_to_id_strict("</tool_call>")?;
        let tool_response = tokenizer.token_to_id_strict("<tool_response>")?;
        let tool_response_end = tokenizer.token_to_id_strict("</tool_response>")?;

        let stop_tokens = vec![im_end, endoftext];

        Ok(Self {
            tokenizer,
            enable_thinking: cfg.enable_thinking,
            preserve_all_thinking: cfg.preserve_all_thinking,
            preserve_thinking_between_tool_calls: cfg.preserve_thinking_between_tool_calls,
            im_start,
            im_end,
            endoftext,
            tool_call,
            tool_call_end,
            tool_response,
            tool_response_end,
            stop_tokens,
        })
    }

    /// Index of the most recent user message whose content is *not* a
    /// `<tool_response>...</tool_response>` placeholder. Defaults to
    /// `len - 1` when no real user message is present.
    fn last_query_index(messages: &[Message]) -> i32 {
        for (i, msg) in messages.iter().enumerate().rev() {
            if msg.role != "user" {
                continue;
            }
            let content = msg.text_content();
            if !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>")) {
                return i as i32;
            }
        }
        (messages.len() as i32).saturating_sub(1)
    }

    fn emit_system_with_tools(
        &self,
        buf: &mut RenderBuf<'_>,
        messages: &[Message],
        tools: &[ToolSpec],
        first_is_system: bool,
    ) -> Result<(), RenderError> {
        let sys_idx: i32 = if first_is_system { 0 } else { SCAFFOLD_IDX };
        buf.special(self.im_start, sys_idx);
        let mut tool_text = String::from("system\n");
        if first_is_system {
            tool_text.push_str(messages[0].text_content());
            tool_text.push_str("\n\n");
        }
        tool_text.push_str(TOOLS_HEADER);
        for tool in tools {
            tool_text.push('\n');
            let spec = tool_spec_template_value(tool);
            tool_text.push_str(&to_string_python(&spec).map_err(|e| {
                RenderError::Invalid(format!("tool spec serialisation failed: {e}"))
            })?);
        }
        tool_text.push_str(TOOLS_FOOTER);
        buf.text(&tool_text, sys_idx)?;
        buf.special(self.im_end, sys_idx);
        buf.text("\n", sys_idx)?;
        Ok(())
    }

    fn emit_system_no_tools(
        &self,
        buf: &mut RenderBuf<'_>,
        messages: &[Message],
    ) -> Result<(), RenderError> {
        buf.special(self.im_start, 0);
        let mut s = String::with_capacity(messages[0].text_content().len() + 8);
        s.push_str("system\n");
        s.push_str(messages[0].text_content());
        buf.text(&s, 0)?;
        buf.special(self.im_end, 0);
        buf.text("\n", 0)?;
        Ok(())
    }

    fn emit_user(
        &self,
        buf: &mut RenderBuf<'_>,
        content: &str,
        idx: i32,
    ) -> Result<(), RenderError> {
        buf.special(self.im_start, idx);
        let mut s = String::with_capacity(content.len() + 8);
        s.push_str("user\n");
        s.push_str(content);
        buf.text(&s, idx)?;
        buf.special(self.im_end, idx);
        buf.text("\n", idx)?;
        Ok(())
    }

    fn emit_non_initial_system(
        &self,
        buf: &mut RenderBuf<'_>,
        content: &str,
        idx: i32,
    ) -> Result<(), RenderError> {
        buf.special(self.im_start, idx);
        let mut s = String::with_capacity(content.len() + 8);
        s.push_str("system\n");
        s.push_str(content);
        buf.text(&s, idx)?;
        buf.special(self.im_end, idx);
        buf.text("\n", idx)?;
        Ok(())
    }

    fn emit_tool(
        &self,
        buf: &mut RenderBuf<'_>,
        messages: &[Message],
        msg_idx: usize,
        content: &str,
    ) -> Result<(), RenderError> {
        let prev_is_tool = msg_idx > 0 && messages[msg_idx - 1].role == "tool";
        let next_is_tool = msg_idx + 1 < messages.len() && messages[msg_idx + 1].role == "tool";
        let idx = msg_idx as i32;

        if !prev_is_tool {
            buf.special(self.im_start, idx);
            buf.text("user", idx)?;
        }
        buf.text("\n", idx)?;
        buf.special(self.tool_response, idx);
        let mut wrapped = String::with_capacity(content.len() + 2);
        wrapped.push('\n');
        wrapped.push_str(content);
        wrapped.push('\n');
        buf.text(&wrapped, idx)?;
        buf.special(self.tool_response_end, idx);
        if !next_is_tool {
            buf.special(self.im_end, idx);
            buf.text("\n", idx)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_assistant(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        msg_idx: usize,
        last_query_index: i32,
        is_last: bool,
        preserve_thinking: bool,
    ) -> Result<(), RenderError> {
        // Recover reasoning content either from the explicit field or
        // from inline `<think>...</think>` text. Match the Python
        // implementation's split semantics exactly.
        let raw_content = msg.text_content();
        let (reasoning_content, content_after_think) = match &msg.reasoning_content {
            Some(s) => (s.clone(), raw_content.to_string()),
            None => {
                if let Some((before, after)) = raw_content.split_once("</think>") {
                    let reasoning = if let Some((_, inner)) = before.rsplit_once("<think>") {
                        inner
                            .trim_start_matches('\n')
                            .trim_end_matches('\n')
                            .to_string()
                    } else {
                        before
                            .trim_start_matches('\n')
                            .trim_end_matches('\n')
                            .to_string()
                    };
                    (reasoning, after.trim_start_matches('\n').to_string())
                } else {
                    (String::new(), raw_content.to_string())
                }
            }
        };

        let idx = msg_idx as i32;
        buf.special(self.im_start, idx);

        let tool_calls = &msg.tool_calls;
        let emit_in_template_window =
            (msg_idx as i32) > last_query_index && (is_last || !reasoning_content.is_empty());
        let emit_via_override = preserve_thinking && !reasoning_content.is_empty();

        let prefix = if emit_in_template_window || emit_via_override {
            let mut s =
                String::with_capacity(reasoning_content.len() + content_after_think.len() + 32);
            s.push_str("assistant\n<think>\n");
            s.push_str(reasoning_content.trim_matches('\n'));
            s.push_str("\n</think>\n\n");
            s.push_str(content_after_think.trim_start_matches('\n'));
            s
        } else {
            let mut s = String::with_capacity(content_after_think.len() + 10);
            s.push_str("assistant\n");
            s.push_str(&content_after_think);
            s
        };

        if tool_calls.is_empty() {
            buf.text(&prefix, idx)?;
        } else {
            for (tc_idx, tc) in tool_calls.iter().enumerate() {
                let name = tc.function.name.as_str();
                let args_str = match &tc.function.arguments {
                    ToolArguments::Raw(s) => s.clone(),
                    ToolArguments::Object(v) => to_string_python(v).map_err(|e| {
                        RenderError::Invalid(format!("tool args serialisation failed: {e}"))
                    })?,
                };
                if tc_idx == 0 {
                    let mut s = prefix.clone();
                    if !content_after_think.is_empty() {
                        s.push('\n');
                    }
                    buf.text(&s, idx)?;
                } else {
                    buf.text("\n", idx)?;
                }
                buf.special(self.tool_call, idx);
                let mut payload = String::with_capacity(args_str.len() + name.len() + 24);
                payload.push_str("\n{\"name\": \"");
                payload.push_str(name);
                payload.push_str("\", \"arguments\": ");
                payload.push_str(&args_str);
                payload.push_str("}\n");
                buf.text(&payload, idx)?;
                buf.special(self.tool_call_end, idx);
            }
        }

        buf.special(self.im_end, idx);
        buf.text("\n", idx)?;
        Ok(())
    }

    fn estimate_capacity(messages: &[Message], tools: Option<&[ToolSpec]>) -> usize {
        // Heuristic: ~256 tokens / message, plus a flat surcharge for the
        // tools block (it can be substantial). Realloc once if we
        // underestimate; the cost of over-allocating is a few KB.
        let base = messages.len().max(1) * 256;
        let tools_bonus = tools.map_or(0, |t| 256 * t.len().max(1));
        base + tools_bonus
    }
}

impl Renderer for Qwen3Renderer {
    fn render(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError> {
        if messages.is_empty() {
            return Err(RenderError::EmptyMessages);
        }
        let cap = Self::estimate_capacity(messages, tools);
        let mut buf = RenderBuf::new(&self.tokenizer, cap);

        let first_is_system = messages[0].role == "system";

        // 1. System + tools header.
        match tools {
            Some(t) if !t.is_empty() => {
                self.emit_system_with_tools(&mut buf, messages, t, first_is_system)?;
            }
            _ => {
                if first_is_system {
                    self.emit_system_no_tools(&mut buf, messages)?;
                }
            }
        }

        // 2. Last-query index.
        let last_qi = Self::last_query_index(messages);
        let num_messages = messages.len();

        // 3. Body.
        for (i, msg) in messages.iter().enumerate() {
            let content = msg.text_content();
            match msg.role.as_str() {
                "system" => {
                    if i == 0 {
                        continue;
                    }
                    self.emit_non_initial_system(&mut buf, content, i as i32)?;
                }
                "user" => {
                    self.emit_user(&mut buf, content, i as i32)?;
                }
                "assistant" => {
                    let preserve_thinking = should_preserve_past_thinking(
                        messages,
                        i,
                        self.preserve_all_thinking,
                        self.preserve_thinking_between_tool_calls,
                    );
                    self.emit_assistant(
                        &mut buf,
                        msg,
                        i,
                        last_qi,
                        i + 1 == num_messages,
                        preserve_thinking,
                    )?;
                }
                "tool" => {
                    self.emit_tool(&mut buf, messages, i, content)?;
                }
                _ => {
                    // Unknown role: skip silently (matches Python which
                    // simply has no branch for it).
                }
            }
        }

        // 4. Generation prompt.
        if add_generation_prompt {
            buf.scaffold_special(self.im_start);
            buf.scaffold_text("assistant\n")?;
            if !self.enable_thinking {
                buf.scaffold_text(GEN_PROMPT_NO_THINKING_SUFFIX)?;
            }
        }

        Ok(buf.into_rendered())
    }

    fn parse_response(&self, token_ids: &[u32]) -> ParsedResponse {
        parse_qwen3(
            &self.tokenizer,
            token_ids,
            &self.stop_tokens,
            self.tool_call,
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

        let cap = Self::estimate_capacity(new_messages, None);
        let mut buf = RenderBuf::new(&self.tokenizer, cap);

        // Trailing `\n` after the prior turn's close token.
        buf.scaffold_text("\n")?;

        for (i, msg) in new_messages.iter().enumerate() {
            let content = msg.text_content();
            let idx = i as i32;
            match msg.role.as_str() {
                "user" => self.emit_user(&mut buf, content, idx)?,
                "system" => self.emit_non_initial_system(&mut buf, content, idx)?,
                "tool" => self.emit_tool(&mut buf, new_messages, i, content)?,
                _ => return Ok(None),
            }
        }

        buf.scaffold_special(self.im_start);
        buf.scaffold_text("assistant\n")?;
        if !self.enable_thinking {
            buf.scaffold_text(GEN_PROMPT_NO_THINKING_SUFFIX)?;
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
