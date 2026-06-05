//! `DeepSeek` V3 renderer. Port of `renderers/deepseek_v3.py`.
//!
//! Key differences from the Qwen-family renderers:
//!
//! - Special tokens use **fullwidth Unicode** delimiters (`｜` = U+FF5C,
//!   `▁` = U+2581). Token names are e.g. `<｜begin▁of▁sentence｜>`.
//! - **Implicit role markers** — `<｜User｜>` and `<｜Assistant｜>` carry the
//!   role themselves; there's no role-name text after the marker the way
//!   Qwen has `<|im_start|>user\n`.
//! - **All leading system messages are concatenated** with `\n\n` and
//!   emitted as plain text *before* the first non-system role token (no
//!   marker for the system block).
//! - Thinking is plain text `<think>...</think>` tags, not special tokens.
//! - Tool calls live in `<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>` with
//!   each call as `<｜tool▁call▁begin｜>function<｜tool▁sep｜>name\n
//!   ` ```json\n{args}\n``` `<｜tool▁call▁end｜>`.

use serde_json::Value as JsonValue;

use crate::bridge::{reject_assistant_in_extension, trim_to_turn_close};
use crate::emit::{RenderBuf, TokenPlanBuf, TokenSink};
use crate::parsing::deepseek_v3::parse_deepseek_v3;
use crate::tokenizer::Tokenizer;
use crate::traits::Renderer;
use crate::types::{Message, ParsedResponse, RenderError, RenderedTokens, ToolArguments, ToolSpec};

const SEP: char = '\u{FF5C}'; // ｜
const US: char = '\u{2581}'; // ▁

fn ds_token(name: &str) -> String {
    let mut s = String::with_capacity(name.len() + 4);
    s.push('<');
    s.push(SEP);
    s.push_str(name);
    s.push(SEP);
    s.push('>');
    s
}

#[derive(Debug, Clone)]
pub struct DeepSeekV3RendererBuilder {
    enable_thinking: bool,
}

impl Default for DeepSeekV3RendererBuilder {
    fn default() -> Self {
        Self {
            enable_thinking: true,
        }
    }
}

impl DeepSeekV3RendererBuilder {
    pub fn enable_thinking(mut self, on: bool) -> Self {
        self.enable_thinking = on;
        self
    }
    pub fn build(self, tokenizer: Tokenizer) -> Result<DeepSeekV3Renderer, RenderError> {
        DeepSeekV3Renderer::new_with(tokenizer, &self)
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekV3Renderer {
    tokenizer: Tokenizer,
    enable_thinking: bool,

    bos: u32,
    eos: u32,
    user_token: u32,
    assistant_token: u32,
    tool_calls_begin: u32,
    tool_calls_end: u32,
    tool_call_begin: u32,
    tool_call_end: u32,
    tool_sep: u32,
    tool_outputs_begin: u32,
    tool_outputs_end: u32,
    tool_output_begin: u32,
    tool_output_end: u32,

    stop_tokens: Vec<u32>,
}

impl DeepSeekV3Renderer {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, RenderError> {
        DeepSeekV3RendererBuilder::default().build(tokenizer)
    }
    pub fn builder() -> DeepSeekV3RendererBuilder {
        DeepSeekV3RendererBuilder::default()
    }

    /// Encode a `DeepSeek` special token via the tokenizer's encode path and
    /// assert it maps to exactly one id. Matches the Python
    /// `_get_special_token` helper — required because the tokenizer
    /// doesn't expose these by `token_to_id` directly (the fullwidth
    /// characters are part of the BPE vocab as a single piece).
    fn resolve(tokenizer: &Tokenizer, name: &str) -> Result<u32, RenderError> {
        let token_str = ds_token(name);
        let encoded = tokenizer.encode_no_special(&token_str)?;
        let ids = encoded.as_slice();
        if ids.len() != 1 {
            return Err(RenderError::MissingSpecialToken(token_str));
        }
        Ok(ids[0])
    }

    // Paired begin/end token ids share semantic prefixes (tool_call,
    // tool_calls, tool_output, tool_outputs); the similarity is the
    // structural relationship, so renaming would lose information.
    #[allow(clippy::similar_names)]
    fn new_with(
        tokenizer: Tokenizer,
        cfg: &DeepSeekV3RendererBuilder,
    ) -> Result<Self, RenderError> {
        let bos = Self::resolve(&tokenizer, &format!("begin{US}of{US}sentence"))?;
        let eos = Self::resolve(&tokenizer, &format!("end{US}of{US}sentence"))?;
        let user_token = Self::resolve(&tokenizer, "User")?;
        let assistant_token = Self::resolve(&tokenizer, "Assistant")?;
        let tool_calls_begin = Self::resolve(&tokenizer, &format!("tool{US}calls{US}begin"))?;
        let tool_calls_end = Self::resolve(&tokenizer, &format!("tool{US}calls{US}end"))?;
        let tool_call_begin = Self::resolve(&tokenizer, &format!("tool{US}call{US}begin"))?;
        let tool_call_end = Self::resolve(&tokenizer, &format!("tool{US}call{US}end"))?;
        let tool_sep = Self::resolve(&tokenizer, &format!("tool{US}sep"))?;
        let tool_outputs_begin = Self::resolve(&tokenizer, &format!("tool{US}outputs{US}begin"))?;
        let tool_outputs_end = Self::resolve(&tokenizer, &format!("tool{US}outputs{US}end"))?;
        let tool_output_begin = Self::resolve(&tokenizer, &format!("tool{US}output{US}begin"))?;
        let tool_output_end = Self::resolve(&tokenizer, &format!("tool{US}output{US}end"))?;

        Ok(Self {
            tokenizer,
            enable_thinking: cfg.enable_thinking,
            bos,
            eos,
            user_token,
            assistant_token,
            tool_calls_begin,
            tool_calls_end,
            tool_call_begin,
            tool_call_end,
            tool_sep,
            tool_outputs_begin,
            tool_outputs_end,
            tool_output_begin,
            tool_output_end,
            stop_tokens: vec![eos],
        })
    }

    fn args_to_json_string(args: &ToolArguments) -> String {
        match args {
            ToolArguments::Raw(s) => s.clone(),
            ToolArguments::Object(v) => python_json_dumps(v),
        }
    }

    fn estimate_capacity(messages: &[Message]) -> usize {
        messages.len().max(1) * 256 + 64
    }

    fn should_batch_encode_text(messages: &[Message], tools: Option<&[ToolSpec]>) -> bool {
        messages.len() >= 8 && tools.is_none_or(<[ToolSpec]>::is_empty)
    }

    fn render_into_buf(
        &self,
        buf: &mut impl TokenSink,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<(), RenderError> {
        if messages.is_empty() {
            return Err(RenderError::EmptyMessages);
        }

        buf.scaffold_special(self.bos);

        let mut first_non_sys = 0usize;
        let mut sys_parts: Vec<&str> = Vec::new();
        for msg in messages {
            if msg.role != "system" {
                break;
            }
            sys_parts.push(msg.text_content());
            first_non_sys += 1;
        }
        if !sys_parts.is_empty() {
            let joined = sys_parts.join("\n\n");
            buf.text(&joined, 0)?;
        }

        for (i, msg) in messages.iter().enumerate().skip(first_non_sys) {
            let idx = i as i32;
            let content = msg.text_content();
            match msg.role.as_str() {
                "system" | "user" => {
                    buf.special(self.user_token, idx);
                    buf.text(content, idx)?;
                }
                "assistant" => self.emit_assistant(buf, msg, i, messages)?,
                "tool" => self.emit_tool(buf, messages, i)?,
                _ => {}
            }
        }

        if add_generation_prompt {
            let last_role = messages.last().map_or("", |m| m.role.as_str());
            if last_role != "tool" {
                buf.scaffold_special(self.assistant_token);
            }
            if self.enable_thinking {
                buf.scaffold_text("<think>\n")?;
            }
        }

        Ok(())
    }
}

fn python_json_dumps(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(v) => v.to_string(),
        JsonValue::Number(v) => v.to_string(),
        JsonValue::String(v) => serde_json::to_string(v).unwrap_or_else(|_| "\"\"".to_string()),
        JsonValue::Array(items) => {
            let mut out = String::from("[");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&python_json_dumps(item));
            }
            out.push(']');
            out
        }
        JsonValue::Object(map) => {
            let mut out = String::from("{");
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string()));
                out.push_str(": ");
                out.push_str(&python_json_dumps(item));
            }
            out.push('}');
            out
        }
    }
}

impl Renderer for DeepSeekV3Renderer {
    fn render(
        &self,
        messages: &[Message],
        _tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError> {
        let mut buf = RenderBuf::new(&self.tokenizer, Self::estimate_capacity(messages));
        self.render_into_buf(&mut buf, messages, add_generation_prompt)?;
        Ok(buf.into_rendered())
    }

    fn render_ids(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>, RenderError> {
        let cap = Self::estimate_capacity(messages);
        if Self::should_batch_encode_text(messages, tools) {
            let mut buf = TokenPlanBuf::new(&self.tokenizer, cap);
            self.render_into_buf(&mut buf, messages, add_generation_prompt)?;
            buf.into_token_ids()
        } else {
            let mut buf = RenderBuf::new_token_ids_only(&self.tokenizer, cap);
            self.render_into_buf(&mut buf, messages, add_generation_prompt)?;
            Ok(buf.into_token_ids())
        }
    }

    fn parse_response(&self, token_ids: &[u32]) -> ParsedResponse {
        parse_deepseek_v3(
            &self.tokenizer,
            token_ids,
            &self.stop_tokens,
            self.tool_calls_begin,
            self.tool_calls_end,
            self.tool_call_begin,
            self.tool_call_end,
            self.tool_sep,
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
            Some(self.eos),
        ) else {
            return Ok(None);
        };

        let mut buf =
            RenderBuf::new_token_ids_only(&self.tokenizer, Self::estimate_capacity(new_messages));

        for (i, msg) in new_messages.iter().enumerate() {
            let idx = i as i32;
            let content = msg.text_content();
            match msg.role.as_str() {
                "user" | "system" => {
                    buf.special(self.user_token, idx);
                    buf.text(content, idx)?;
                }
                "tool" => {
                    let prev_is_tool = i > 0 && new_messages[i - 1].role == "tool";
                    let next_is_tool =
                        i + 1 < new_messages.len() && new_messages[i + 1].role == "tool";
                    if !prev_is_tool {
                        buf.special(self.tool_outputs_begin, idx);
                    }
                    buf.special(self.tool_output_begin, idx);
                    buf.text(content, idx)?;
                    buf.special(self.tool_output_end, idx);
                    if !next_is_tool {
                        buf.special(self.tool_outputs_end, idx);
                    }
                }
                _ => return Ok(None),
            }
        }

        let last_role = new_messages.last().map_or("", |m| m.role.as_str());
        if last_role != "tool" {
            buf.scaffold_special(self.assistant_token);
        }
        if self.enable_thinking {
            buf.scaffold_text("<think>\n")?;
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

impl DeepSeekV3Renderer {
    fn emit_assistant(
        &self,
        buf: &mut impl TokenSink,
        msg: &Message,
        msg_idx: usize,
        messages: &[Message],
    ) -> Result<(), RenderError> {
        let prev_is_tool = msg_idx > 0 && messages[msg_idx - 1].role == "tool";
        let idx = msg_idx as i32;

        // Build the content text, with reasoning_content wrapped in <think> if present
        let mut content = msg.text_content().to_string();
        if let Some(reasoning) = msg.reasoning_content.as_deref() {
            if !reasoning.is_empty() {
                let mut wrapped = String::with_capacity(reasoning.len() + content.len() + 16);
                wrapped.push_str("<think>");
                wrapped.push_str(reasoning);
                wrapped.push_str("</think>");
                wrapped.push_str(&content);
                content = wrapped;
            }
        }

        if !prev_is_tool {
            buf.special(self.assistant_token, idx);
        }

        // Pre-tool-call content
        buf.text(&content, idx)?;

        if !msg.tool_calls.is_empty() {
            buf.special(self.tool_calls_begin, idx);
            for tc in &msg.tool_calls {
                let name = tc.function.name.as_str();
                let args_str = Self::args_to_json_string(&tc.function.arguments);

                buf.special(self.tool_call_begin, idx);
                buf.text("function", idx)?;
                buf.special(self.tool_sep, idx);
                let mut payload = String::with_capacity(name.len() + args_str.len() + 16);
                payload.push_str(name);
                payload.push_str("\n```json\n");
                payload.push_str(&args_str);
                payload.push_str("\n```");
                buf.text(&payload, idx)?;
                buf.special(self.tool_call_end, idx);
            }
            buf.special(self.tool_calls_end, idx);
        }

        buf.special(self.eos, idx);
        Ok(())
    }

    fn emit_tool(
        &self,
        buf: &mut impl TokenSink,
        messages: &[Message],
        msg_idx: usize,
    ) -> Result<(), RenderError> {
        let prev_is_tool = msg_idx > 0 && messages[msg_idx - 1].role == "tool";
        let next_is_tool = msg_idx + 1 < messages.len() && messages[msg_idx + 1].role == "tool";
        let idx = msg_idx as i32;
        let content = messages[msg_idx].text_content();

        if !prev_is_tool {
            buf.special(self.tool_outputs_begin, idx);
        }
        buf.special(self.tool_output_begin, idx);
        buf.text(content, idx)?;
        buf.special(self.tool_output_end, idx);
        if !next_is_tool {
            buf.special(self.tool_outputs_end, idx);
        }
        Ok(())
    }
}
