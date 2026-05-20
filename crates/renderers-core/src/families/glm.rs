//! GLM family renderers — covers GLM-5, GLM-5.1, and GLM-4.5 Air.
//!
//! Port of `renderers/glm5.py` (+ `GLM51Renderer`) and `renderers/glm45.py`.
//!
//! Shared template shape:
//!
//! - Prefix: `[gMASK]<sop>` before all content
//! - Role markers: `<|system|>`, `<|user|>`, `<|assistant|>`,
//!   `<|observation|>`. No role-name text follows the marker.
//! - **No close token** — turns end when the next role marker appears.
//!   `bridge_to_next_turn` exploits this: the prior turn's tail
//!   contains one of `{<|endoftext|>, <|user|>, <|observation|>}`
//!   (the stop ids), so the bridge synthesises `<|endoftext|>` only on
//!   truncation.
//! - Tool calls: `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`
//!
//! Variants in this module:
//!
//! | Flag                          | GLM-5 | GLM-5.1 | GLM-4.5 |
//! | ----------------------------- | ----- | ------- | ------- |
//! | newlines after role markers   | no    | no      | yes     |
//! | newlines inside tool-call     | no    | no      | yes     |
//! | `/nothink` user suffix        | no    | no      | yes     |
//! | empty `<think></think>` wrap  | no    | yes     | no      |
//! | unwrap OpenAI tool envelope   | no    | yes     | no      |
//!
//! The flags are surfaced on the builder; the three variants pick
//! their own combination at construction time.

use serde_json::Value as JsonValue;

use crate::bridge::reject_assistant_in_extension;
use crate::emit::RenderBuf;
use crate::parsing::glm::parse_glm;
use crate::thinking::should_preserve_past_thinking;
use crate::tokenizer::Tokenizer;
use crate::traits::Renderer;
use crate::types::{
    Message, ParsedResponse, RenderError, RenderedTokens, ToolArguments, ToolSpec, SCAFFOLD_IDX,
};

const TOOLS_HEADER_GLM5: &str = "\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
const TOOLS_FOOTER_GLM5: &str = "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";

const TOOLS_FOOTER_GLM45: &str = "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variant {
    Glm5,
    Glm51,
    Glm45,
}

#[derive(Debug, Clone)]
pub struct GlmRendererBuilder {
    enable_thinking: bool,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,
    variant: Variant,
}

impl GlmRendererBuilder {
    pub fn glm5() -> Self {
        Self {
            enable_thinking: true,
            preserve_all_thinking: false,
            preserve_thinking_between_tool_calls: false,
            variant: Variant::Glm5,
        }
    }
    pub fn glm51() -> Self {
        Self { variant: Variant::Glm51, ..Self::glm5() }
    }
    pub fn glm45() -> Self {
        Self { variant: Variant::Glm45, ..Self::glm5() }
    }
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
    pub fn build(self, tokenizer: Tokenizer) -> Result<GlmRenderer, RenderError> {
        GlmRenderer::new_with(tokenizer, self)
    }
}

#[derive(Debug, Clone)]
pub struct GlmRenderer {
    tokenizer: Tokenizer,
    variant: Variant,
    enable_thinking: bool,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,

    gmask: u32,
    sop: u32,
    system: u32,
    user: u32,
    assistant: u32,
    observation: u32,
    endoftext: u32,
    think: u32,
    think_end: u32,
    tool_call: u32,
    tool_call_end: u32,
    arg_key: u32,
    arg_key_end: u32,
    arg_value: u32,
    arg_value_end: u32,
    // GLM-5 also exposes <tool_response> tokens; GLM-4.5 emits them as text.
    tool_response: Option<u32>,
    tool_response_end: Option<u32>,

    stop_tokens: Vec<u32>,
}

impl GlmRenderer {
    pub fn glm5(tokenizer: Tokenizer) -> Result<Self, RenderError> {
        GlmRendererBuilder::glm5().build(tokenizer)
    }
    pub fn glm51(tokenizer: Tokenizer) -> Result<Self, RenderError> {
        GlmRendererBuilder::glm51().build(tokenizer)
    }
    pub fn glm45(tokenizer: Tokenizer) -> Result<Self, RenderError> {
        GlmRendererBuilder::glm45().build(tokenizer)
    }

    fn new_with(tokenizer: Tokenizer, cfg: GlmRendererBuilder) -> Result<Self, RenderError> {
        let gmask = tokenizer.token_to_id_strict("[gMASK]")?;
        let sop = tokenizer.token_to_id_strict("<sop>")?;
        let system = tokenizer.token_to_id_strict("<|system|>")?;
        let user = tokenizer.token_to_id_strict("<|user|>")?;
        let assistant = tokenizer.token_to_id_strict("<|assistant|>")?;
        let observation = tokenizer.token_to_id_strict("<|observation|>")?;
        let endoftext = tokenizer.token_to_id_strict("<|endoftext|>")?;
        let think = tokenizer.token_to_id_strict("<think>")?;
        let think_end = tokenizer.token_to_id_strict("</think>")?;
        let tool_call = tokenizer.token_to_id_strict("<tool_call>")?;
        let tool_call_end = tokenizer.token_to_id_strict("</tool_call>")?;
        let arg_key = tokenizer.token_to_id_strict("<arg_key>")?;
        let arg_key_end = tokenizer.token_to_id_strict("</arg_key>")?;
        let arg_value = tokenizer.token_to_id_strict("<arg_value>")?;
        let arg_value_end = tokenizer.token_to_id_strict("</arg_value>")?;

        // GLM-5 uses <tool_response> special tokens; GLM-4.5 emits them
        // as plain text. Resolve optionally so the same struct serves
        // both variants.
        let (tool_response, tool_response_end) = if cfg.variant == Variant::Glm45 {
            (None, None)
        } else {
            (
                Some(tokenizer.token_to_id_strict("<tool_response>")?),
                Some(tokenizer.token_to_id_strict("</tool_response>")?),
            )
        };

        Ok(Self {
            tokenizer,
            variant: cfg.variant,
            enable_thinking: cfg.enable_thinking,
            preserve_all_thinking: cfg.preserve_all_thinking,
            preserve_thinking_between_tool_calls: cfg.preserve_thinking_between_tool_calls,
            gmask,
            sop,
            system,
            user,
            assistant,
            observation,
            endoftext,
            think,
            think_end,
            tool_call,
            tool_call_end,
            arg_key,
            arg_key_end,
            arg_value,
            arg_value_end,
            tool_response,
            tool_response_end,
            stop_tokens: vec![endoftext, user, observation],
        })
    }

    fn nl_after_role(&self) -> &'static str {
        if self.variant == Variant::Glm45 { "\n" } else { "" }
    }

    fn empty_think_on_last_assistant(&self) -> bool {
        self.variant == Variant::Glm51
    }

    fn last_user_index(messages: &[Message]) -> i32 {
        for (i, m) in messages.iter().enumerate().rev() {
            if m.role == "user" {
                return i as i32;
            }
        }
        -1
    }

    fn format_tool_spec(&self, tool: &ToolSpec) -> Result<String, RenderError> {
        // GLM-5 / GLM-4.5 render the spec verbatim; GLM-5.1 unwraps the
        // OpenAI envelope (`{"type":"function","function":{...}}`) and
        // strips internal-only keys.
        //
        // Our `ToolSpec` is already the inner shape, so the GLM-5.1
        // unwrap is a no-op in Rust — kept here as a structural note.
        let spec = serde_json::json!({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        });
        serde_json::to_string(&spec)
            .map_err(|e| RenderError::Invalid(format!("tool spec serialisation: {e}")))
    }

    fn render_arg_value(arg_value: &JsonValue) -> String {
        match arg_value {
            JsonValue::String(s) => s.clone(),
            _ => serde_json::to_string(arg_value).unwrap_or_default(),
        }
    }
}

impl Renderer for GlmRenderer {
    fn render(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError> {
        if messages.is_empty() {
            return Err(RenderError::EmptyMessages);
        }
        let nl = self.nl_after_role();
        let mut buf = RenderBuf::new(
            &self.tokenizer,
            messages.len().max(1) * 256 + tools.map(|t| t.len() * 256 + 256).unwrap_or(0),
        );

        // Prefix
        buf.scaffold_special(self.gmask);
        buf.scaffold_special(self.sop);

        // Tools system block
        if let Some(t) = tools {
            if !t.is_empty() {
                buf.scaffold_special(self.system);
                let mut s = String::with_capacity(512);
                if !nl.is_empty() {
                    s.push_str(nl);
                }
                s.push_str(TOOLS_HEADER_GLM5);
                for tool in t {
                    s.push_str(&self.format_tool_spec(tool)?);
                    s.push('\n');
                }
                s.push_str(if self.variant == Variant::Glm45 {
                    TOOLS_FOOTER_GLM45
                } else {
                    TOOLS_FOOTER_GLM5
                });
                buf.scaffold_text(&s)?;
            }
        }

        let last_ui = Self::last_user_index(messages);

        for (i, msg) in messages.iter().enumerate() {
            let content = msg.text_content();
            let idx = i as i32;
            match msg.role.as_str() {
                "system" => {
                    buf.special(self.system, idx);
                    let mut s = String::with_capacity(content.len() + 2);
                    s.push_str(nl);
                    s.push_str(content);
                    buf.text(&s, idx)?;
                }
                "user" => {
                    buf.special(self.user, idx);
                    let mut s = String::with_capacity(content.len() + 12);
                    s.push_str(nl);
                    s.push_str(content);
                    if self.variant == Variant::Glm45
                        && !self.enable_thinking
                        && !content.ends_with("/nothink")
                    {
                        s.push_str("/nothink");
                    }
                    buf.text(&s, idx)?;
                }
                "assistant" => {
                    let preserve_thinking = should_preserve_past_thinking(
                        messages,
                        i,
                        self.preserve_all_thinking,
                        self.preserve_thinking_between_tool_calls,
                    );
                    self.emit_assistant(&mut buf, msg, idx, last_ui, preserve_thinking)?;
                }
                "tool" => self.emit_tool(&mut buf, messages, i, content, idx)?,
                _ => {} // mirror Python: silent skip
            }
        }

        if add_generation_prompt {
            buf.scaffold_special(self.assistant);
            if self.variant == Variant::Glm45 {
                if !self.enable_thinking {
                    buf.scaffold_text("\n")?;
                    buf.scaffold_special(self.think);
                    buf.scaffold_special(self.think_end);
                }
                // GLM-4.5 enable_thinking=True: just <|assistant|>, nothing else
            } else if self.enable_thinking {
                buf.scaffold_special(self.think);
            } else {
                buf.scaffold_special(self.think_end);
            }
        }

        Ok(buf.into_rendered())
    }

    fn parse_response(&self, token_ids: &[u32]) -> ParsedResponse {
        parse_glm(
            &self.tokenizer,
            token_ids,
            &self.stop_tokens,
            self.think,
            self.think_end,
            self.tool_call,
            self.tool_call_end,
            self.arg_key,
            self.arg_key_end,
            self.arg_value,
            self.arg_value_end,
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

        // GLM has no per-turn close token. Build the combined prefix and
        // synthesise <|endoftext|> when the model's completion ran past
        // max_tokens (no stop-id at the tail).
        let mut combined: Vec<u32> =
            Vec::with_capacity(previous_prompt_ids.len() + previous_completion_ids.len() + 1);
        combined.extend_from_slice(previous_prompt_ids);
        combined.extend_from_slice(previous_completion_ids);

        let need_synth = match combined.last() {
            None => true,
            Some(&t) if !self.stop_tokens.contains(&t) => true,
            _ => previous_completion_ids.is_empty(),
        };
        if need_synth {
            combined.push(self.endoftext);
        }
        let last_prev = *combined.last().expect("non-empty");

        let nl = self.nl_after_role();
        let mut buf = RenderBuf::new(
            &self.tokenizer,
            new_messages.len().max(1) * 256,
        );

        for (i, msg) in new_messages.iter().enumerate() {
            let idx = i as i32;
            let content = msg.text_content();
            match msg.role.as_str() {
                "user" => {
                    if !(i == 0 && last_prev == self.user) {
                        buf.special(self.user, idx);
                    }
                    let mut s = String::with_capacity(content.len() + 12);
                    s.push_str(nl);
                    s.push_str(content);
                    if self.variant == Variant::Glm45
                        && !self.enable_thinking
                        && !content.ends_with("/nothink")
                    {
                        s.push_str("/nothink");
                    }
                    buf.text(&s, idx)?;
                }
                "system" => {
                    buf.special(self.system, idx);
                    let mut s = String::with_capacity(content.len() + 2);
                    s.push_str(nl);
                    s.push_str(content);
                    buf.text(&s, idx)?;
                }
                "tool" => {
                    let prev_is_tool = i > 0 && new_messages[i - 1].role == "tool";
                    if i == 0 && last_prev == self.observation {
                        // model already emitted the marker; don't repeat
                    } else if !prev_is_tool {
                        buf.special(self.observation, idx);
                    }
                    self.emit_tool_response(&mut buf, content, idx)?;
                }
                _ => return Ok(None),
            }
        }

        // Generation prompt
        buf.scaffold_special(self.assistant);
        if self.variant == Variant::Glm45 {
            if !self.enable_thinking {
                buf.scaffold_text("\n")?;
                buf.scaffold_special(self.think);
                buf.scaffold_special(self.think_end);
            }
        } else if self.enable_thinking {
            buf.scaffold_special(self.think);
        } else {
            buf.scaffold_special(self.think_end);
        }

        let ext = buf.into_token_ids();
        let mut out = Vec::with_capacity(combined.len() + ext.len());
        out.extend_from_slice(&combined);
        out.extend_from_slice(&ext);
        Ok(Some(RenderedTokens {
            token_ids: out,
            message_indices: Vec::new(),
            multi_modal_data: None,
        }))
    }
}

impl GlmRenderer {
    fn emit_assistant(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        msg_idx: i32,
        last_user_index: i32,
        preserve_thinking: bool,
    ) -> Result<(), RenderError> {
        let raw_content = msg.text_content();
        let (reasoning_content, content) = match &msg.reasoning_content {
            Some(s) => (s.clone(), raw_content.to_string()),
            None => {
                if let Some((before, after)) = raw_content.split_once("</think>") {
                    let r = if let Some((_, inner)) = before.rsplit_once("<think>") {
                        inner.trim_start_matches('\n').trim_end_matches('\n').to_string()
                    } else {
                        before.trim_start_matches('\n').trim_end_matches('\n').to_string()
                    };
                    (r, after.trim_start_matches('\n').to_string())
                } else {
                    (String::new(), raw_content.to_string())
                }
            }
        };
        let reasoning_content = reasoning_content.trim().to_string();
        let content = content.trim().to_string();

        buf.special(self.assistant, msg_idx);

        if self.variant == Variant::Glm45 {
            self.emit_assistant_glm45(buf, msg, msg_idx, &reasoning_content, &content, last_user_index, preserve_thinking)
        } else {
            self.emit_assistant_glm5_family(buf, msg, msg_idx, &reasoning_content, &content, last_user_index, preserve_thinking)
        }
    }

    fn emit_assistant_glm5_family(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        msg_idx: i32,
        reasoning_content: &str,
        content: &str,
        last_user_index: i32,
        preserve_thinking: bool,
    ) -> Result<(), RenderError> {
        let include_thinking =
            (msg_idx > last_user_index || preserve_thinking) && !reasoning_content.is_empty();
        if include_thinking {
            buf.special(self.think, msg_idx);
            buf.text(reasoning_content.trim(), msg_idx)?;
            buf.special(self.think_end, msg_idx);
        } else if self.empty_think_on_last_assistant() && msg_idx > last_user_index {
            // GLM-5.1: wrap the last assistant with empty <think></think>
            buf.special(self.think, msg_idx);
            buf.special(self.think_end, msg_idx);
        } else {
            buf.special(self.think_end, msg_idx);
        }

        if !content.trim().is_empty() {
            buf.text(content.trim(), msg_idx)?;
        }

        for tc in &msg.tool_calls {
            let name = tc.function.name.as_str();
            buf.special(self.tool_call, msg_idx);
            buf.text(name, msg_idx)?;
            let args_value = match &tc.function.arguments {
                ToolArguments::Object(v) => v.clone(),
                ToolArguments::Raw(s) => serde_json::from_str(s).unwrap_or(JsonValue::Object(Default::default())),
            };
            if let Some(obj) = args_value.as_object() {
                for (k, v) in obj {
                    buf.special(self.arg_key, msg_idx);
                    buf.text(k, msg_idx)?;
                    buf.special(self.arg_key_end, msg_idx);
                    buf.special(self.arg_value, msg_idx);
                    buf.text(&Self::render_arg_value(v), msg_idx)?;
                    buf.special(self.arg_value_end, msg_idx);
                }
            }
            buf.special(self.tool_call_end, msg_idx);
        }
        Ok(())
    }

    fn emit_assistant_glm45(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        msg_idx: i32,
        reasoning_content: &str,
        content: &str,
        last_user_index: i32,
        preserve_thinking: bool,
    ) -> Result<(), RenderError> {
        if (msg_idx > last_user_index || preserve_thinking) && !reasoning_content.is_empty() {
            buf.text("\n", msg_idx)?;
            buf.special(self.think, msg_idx);
            buf.text(reasoning_content.trim(), msg_idx)?;
            buf.special(self.think_end, msg_idx);
        } else {
            buf.text("\n", msg_idx)?;
            buf.special(self.think, msg_idx);
            buf.special(self.think_end, msg_idx);
        }

        let tool_calls = &msg.tool_calls;
        let trimmed = content.trim();
        if !trimmed.is_empty() && !tool_calls.is_empty() {
            let mut s = String::with_capacity(trimmed.len() + 2);
            s.push('\n');
            s.push_str(trimmed);
            s.push('\n');
            buf.text(&s, msg_idx)?;
        } else if !trimmed.is_empty() {
            let mut s = String::with_capacity(trimmed.len() + 1);
            s.push('\n');
            s.push_str(trimmed);
            buf.text(&s, msg_idx)?;
        }

        for tc in tool_calls {
            let name = tc.function.name.as_str();
            if trimmed.is_empty() {
                buf.text("\n", msg_idx)?;
            }
            buf.special(self.tool_call, msg_idx);
            let mut head = String::with_capacity(name.len() + 1);
            head.push_str(name);
            head.push('\n');
            buf.text(&head, msg_idx)?;

            let args_value = match &tc.function.arguments {
                ToolArguments::Object(v) => v.clone(),
                ToolArguments::Raw(s) => serde_json::from_str(s).unwrap_or(JsonValue::Object(Default::default())),
            };
            if let Some(obj) = args_value.as_object() {
                for (k, v) in obj {
                    buf.special(self.arg_key, msg_idx);
                    buf.text(k, msg_idx)?;
                    buf.special(self.arg_key_end, msg_idx);
                    buf.text("\n", msg_idx)?;
                    buf.special(self.arg_value, msg_idx);
                    buf.text(&Self::render_arg_value(v), msg_idx)?;
                    buf.special(self.arg_value_end, msg_idx);
                    buf.text("\n", msg_idx)?;
                }
            }
            buf.special(self.tool_call_end, msg_idx);
        }
        Ok(())
    }

    fn emit_tool(
        &self,
        buf: &mut RenderBuf<'_>,
        messages: &[Message],
        msg_idx: usize,
        content: &str,
        idx: i32,
    ) -> Result<(), RenderError> {
        let prev_is_tool = msg_idx > 0 && messages[msg_idx - 1].role == "tool";
        if !prev_is_tool {
            buf.special(self.observation, idx);
        }
        self.emit_tool_response(buf, content, idx)
    }

    fn emit_tool_response(
        &self,
        buf: &mut RenderBuf<'_>,
        content: &str,
        idx: i32,
    ) -> Result<(), RenderError> {
        if self.variant == Variant::Glm45 {
            // GLM-4.5 emits the tool_response wrapper as plain text
            let mut s = String::with_capacity(content.len() + 32);
            s.push_str("\n<tool_response>\n");
            s.push_str(content);
            s.push_str("\n</tool_response>");
            buf.text(&s, idx)?;
        } else {
            // GLM-5 / GLM-5.1 use special tokens
            buf.special(self.tool_response.expect("tool_response token"), idx);
            buf.text(content, idx)?;
            buf.special(self.tool_response_end.expect("tool_response_end token"), idx);
        }
        Ok(())
    }
}

// Kept for completeness; GLM-5 doesn't ship the `<|endoftext|>` flag the
// way Nemotron does, so the field is always Some.
#[allow(dead_code)]
fn _glm_invariants() {
    let _ = SCAFFOLD_IDX;
}
