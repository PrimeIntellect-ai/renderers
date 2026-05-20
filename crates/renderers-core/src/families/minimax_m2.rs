//! MiniMax M2.5 renderer. Port of `renderers/minimax_m2.py`.
//!
//! Unique characteristics:
//!
//! - Token format: `]~!b[` (BOS), `]~b]` (role prefix), `[e~[` (EOS).
//!   Role "assistant" is rendered as "ai".
//! - System block always present — default system message
//!   ("You are a helpful assistant. Your name is MiniMax-M2.5 and is
//!   built by MiniMax.") auto-injected if missing.
//! - Tools, when supplied, are appended to the system message as
//!   `<tool>{json}</tool>` lines inside a `<tools>...</tools>` block,
//!   followed by a verbose instructions block.
//! - Tool calls use XML wrapper + nested invokes:
//!   `<minimax:tool_call><invoke name="fn"><parameter name="k">v</parameter>...
//!   </invoke></minimax:tool_call>`
//! - Tool responses wrapped in literal `<response>...</response>`
//!   (plain text, no special token).
//! - Thinking emitted only for assistants after the last user turn
//!   (or when preserve_all_thinking is on).

use crate::bridge::{reject_assistant_in_extension, trim_to_turn_close};
use crate::emit::RenderBuf;
use crate::json::to_string_python;
use crate::parsing::minimax::parse_minimax;
use crate::thinking::should_preserve_past_thinking;
use crate::tokenizer::Tokenizer;
use crate::traits::Renderer;
use crate::types::{
    Message, ParsedResponse, RenderError, RenderedTokens, SCAFFOLD_IDX, ToolArguments, ToolSpec,
};

const DEFAULT_SYSTEM: &str =
    "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax.";

const TOOLS_HEADER: &str = "\n\n# Tools\nYou may call one or more tools to assist with the user query.\nHere are the tools available in JSONSchema format:\n\n<tools>\n";
const TOOLS_FOOTER_PREFIX: &str = "</tools>\n\n";
const TOOLS_INSTRUCTIONS: &str = "When making tool calls, use XML format to invoke tools and pass parameters:\n\n<minimax:tool_call>\n<invoke name=\"tool-name-1\">\n<parameter name=\"param-key-1\">param-value-1</parameter>\n<parameter name=\"param-key-2\">param-value-2</parameter>\n...\n</invoke>\n</minimax:tool_call>";

#[derive(Debug, Clone)]
pub struct MiniMaxM2RendererBuilder {
    default_system: String,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,
}

impl Default for MiniMaxM2RendererBuilder {
    fn default() -> Self {
        Self {
            default_system: DEFAULT_SYSTEM.to_string(),
            preserve_all_thinking: false,
            preserve_thinking_between_tool_calls: false,
        }
    }
}

impl MiniMaxM2RendererBuilder {
    pub fn default_system(mut self, s: impl Into<String>) -> Self {
        self.default_system = s.into();
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
    pub fn build(self, tokenizer: Tokenizer) -> Result<MiniMaxM2Renderer, RenderError> {
        MiniMaxM2Renderer::new_with(tokenizer, self)
    }
}

#[derive(Debug, Clone)]
pub struct MiniMaxM2Renderer {
    tokenizer: Tokenizer,
    default_system: String,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,

    bos: u32,
    role: u32,
    eos: u32,
    think: u32,
    think_end: u32,
    tool_call: u32,
    tool_call_end: u32,

    stop_tokens: Vec<u32>,
}

impl MiniMaxM2Renderer {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, RenderError> {
        MiniMaxM2RendererBuilder::default().build(tokenizer)
    }
    pub fn builder() -> MiniMaxM2RendererBuilder {
        MiniMaxM2RendererBuilder::default()
    }

    fn new_with(tokenizer: Tokenizer, cfg: MiniMaxM2RendererBuilder) -> Result<Self, RenderError> {
        let bos = tokenizer.token_to_id_strict("]~!b[")?;
        let role = tokenizer.token_to_id_strict("]~b]")?;
        let eos = tokenizer.token_to_id_strict("[e~[")?;
        let think = tokenizer.token_to_id_strict("<think>")?;
        let think_end = tokenizer.token_to_id_strict("</think>")?;
        let tool_call = tokenizer.token_to_id_strict("<minimax:tool_call>")?;
        let tool_call_end = tokenizer.token_to_id_strict("</minimax:tool_call>")?;

        Ok(Self {
            tokenizer,
            default_system: cfg.default_system,
            preserve_all_thinking: cfg.preserve_all_thinking,
            preserve_thinking_between_tool_calls: cfg.preserve_thinking_between_tool_calls,
            bos,
            role,
            eos,
            think,
            think_end,
            tool_call,
            tool_call_end,
            stop_tokens: vec![eos],
        })
    }

    fn build_system_text(&self, sys_content: &str, tools: Option<&[ToolSpec]>) -> String {
        let mut s = String::with_capacity(512);
        s.push_str("system\n");
        if sys_content.is_empty() {
            s.push_str(&self.default_system);
        } else {
            s.push_str(sys_content);
        }
        if let Some(tools) = tools {
            if !tools.is_empty() {
                s.push_str(TOOLS_HEADER);
                for tool in tools {
                    s.push_str("<tool>");
                    let spec = serde_json::json!({
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    });
                    s.push_str(&to_string_python(&spec).unwrap_or_default());
                    s.push_str("</tool>\n");
                }
                s.push_str(TOOLS_FOOTER_PREFIX);
                s.push_str(TOOLS_INSTRUCTIONS);
            }
        }
        s
    }

    fn args_to_value(args: &ToolArguments) -> serde_json::Value {
        match args {
            ToolArguments::Object(v) => v.clone(),
            ToolArguments::Raw(s) => {
                serde_json::from_str(s).unwrap_or(serde_json::Value::Object(Default::default()))
            }
        }
    }
}

impl Renderer for MiniMaxM2Renderer {
    fn render(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError> {
        if messages.is_empty() {
            return Err(RenderError::EmptyMessages);
        }
        let mut buf = RenderBuf::new(
            &self.tokenizer,
            messages.len().max(1) * 256 + tools.map(|t| t.len() * 256 + 512).unwrap_or(0),
        );

        let first_is_system = messages[0].role == "system";
        let sys_idx: i32 = if first_is_system { 0 } else { SCAFFOLD_IDX };

        // System block
        buf.special(self.bos, sys_idx);
        buf.special(self.role, sys_idx);
        let sys_content = if first_is_system {
            messages[0].visible_text_content().to_string()
        } else {
            String::new()
        };
        let system_text = self.build_system_text(&sys_content, tools);
        buf.text(&system_text, sys_idx)?;
        buf.special(self.eos, sys_idx);
        buf.text("\n", sys_idx)?;

        // Conversation messages — skip the leading system if present
        let conversation_start = usize::from(first_is_system);
        let conversation = &messages[conversation_start..];

        // last_user_index relative to the conversation
        let mut last_ui: i32 = -1;
        for (ci, m) in conversation.iter().enumerate() {
            if m.role == "user" {
                last_ui = ci as i32;
            }
        }

        for (ci, msg) in conversation.iter().enumerate() {
            let orig_idx = (ci + conversation_start) as i32;
            let content = msg.visible_text_content();
            match msg.role.as_str() {
                "user" => {
                    buf.special(self.role, orig_idx);
                    let mut s = String::with_capacity(content.len() + 8);
                    s.push_str("user\n");
                    s.push_str(content);
                    buf.text(&s, orig_idx)?;
                    buf.special(self.eos, orig_idx);
                    buf.text("\n", orig_idx)?;
                }
                "assistant" => {
                    let preserve_thinking = should_preserve_past_thinking(
                        messages,
                        orig_idx as usize,
                        self.preserve_all_thinking,
                        self.preserve_thinking_between_tool_calls,
                    );
                    self.emit_assistant(
                        &mut buf,
                        msg,
                        orig_idx,
                        ci as i32,
                        last_ui,
                        preserve_thinking,
                    )?;
                }
                "tool" => self.emit_tool(&mut buf, conversation, ci, orig_idx)?,
                _ => {}
            }
        }

        if add_generation_prompt {
            buf.scaffold_special(self.role);
            buf.scaffold_text("ai\n")?;
            buf.scaffold_special(self.think);
            buf.scaffold_text("\n")?;
        }

        Ok(buf.into_rendered())
    }

    fn parse_response(&self, token_ids: &[u32]) -> ParsedResponse {
        parse_minimax(
            &self.tokenizer,
            token_ids,
            &self.stop_tokens,
            self.think,
            self.think_end,
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
            Some(self.eos),
        ) else {
            return Ok(None);
        };

        let mut buf = RenderBuf::new(&self.tokenizer, new_messages.len().max(1) * 256);
        // Trailing \n after the prior turn's [e~[
        buf.scaffold_text("\n")?;

        for (i, msg) in new_messages.iter().enumerate() {
            let idx = i as i32;
            let content = msg.visible_text_content();
            match msg.role.as_str() {
                "user" => {
                    buf.special(self.role, idx);
                    let mut s = String::with_capacity(content.len() + 8);
                    s.push_str("user\n");
                    s.push_str(content);
                    buf.text(&s, idx)?;
                    buf.special(self.eos, idx);
                    buf.text("\n", idx)?;
                }
                "system" => {
                    buf.special(self.role, idx);
                    let mut s = String::with_capacity(content.len() + 8);
                    s.push_str("system\n");
                    s.push_str(content);
                    buf.text(&s, idx)?;
                    buf.special(self.eos, idx);
                    buf.text("\n", idx)?;
                }
                "tool" => self.emit_tool(&mut buf, new_messages, i, idx)?,
                _ => return Ok(None),
            }
        }

        buf.scaffold_special(self.role);
        buf.scaffold_text("ai\n")?;
        buf.scaffold_special(self.think);
        buf.scaffold_text("\n")?;

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

impl MiniMaxM2Renderer {
    fn emit_assistant(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        orig_idx: i32,
        conv_idx: i32,
        last_user_index: i32,
        preserve_thinking: bool,
    ) -> Result<(), RenderError> {
        let raw_content = msg.visible_text_content();
        let (reasoning_content, content_text) = match &msg.reasoning_content {
            Some(s) => (s.clone(), raw_content.to_string()),
            None => {
                if let Some((before, after)) = raw_content.split_once("</think>") {
                    let r = if let Some((_, inner)) = before.rsplit_once("<think>") {
                        inner.trim_matches('\n').to_string()
                    } else {
                        before.trim_matches('\n').to_string()
                    };
                    (r, after.trim_matches('\n').to_string())
                } else {
                    (String::new(), raw_content.to_string())
                }
            }
        };

        buf.special(self.role, orig_idx);

        let tool_calls = &msg.tool_calls;
        let emit_think =
            !reasoning_content.is_empty() && (conv_idx > last_user_index || preserve_thinking);

        let after_think: String = if emit_think {
            buf.text("ai\n", orig_idx)?;
            buf.special(self.think, orig_idx);
            let mut head = String::with_capacity(reasoning_content.len() + 2);
            head.push('\n');
            head.push_str(&reasoning_content);
            head.push('\n');
            buf.text(&head, orig_idx)?;
            buf.special(self.think_end, orig_idx);
            // After </think>, the rest is "\n\n" + content (or just "\n\n")
            if content_text.is_empty() {
                "\n\n".to_string()
            } else {
                let mut s = String::with_capacity(content_text.len() + 2);
                s.push_str("\n\n");
                s.push_str(&content_text);
                s
            }
        } else if content_text.is_empty() {
            "ai\n".to_string()
        } else {
            let mut s = String::with_capacity(content_text.len() + 4);
            s.push_str("ai\n");
            s.push_str(&content_text);
            s
        };

        if tool_calls.is_empty() {
            buf.text(&after_think, orig_idx)?;
        } else {
            // \n before <minimax:tool_call> contiguous with preceding text
            let mut head = after_think;
            head.push('\n');
            buf.text(&head, orig_idx)?;
            buf.special(self.tool_call, orig_idx);

            let mut invoke_block = String::from("\n");
            for tc in tool_calls {
                let name = tc.function.name.as_str();
                invoke_block.push_str("<invoke name=\"");
                invoke_block.push_str(name);
                invoke_block.push_str("\">\n");
                let args_value = Self::args_to_value(&tc.function.arguments);
                if let Some(obj) = args_value.as_object() {
                    for (arg_name, arg_value) in obj {
                        let val_str = match arg_value {
                            serde_json::Value::String(s) => s.clone(),
                            _ => serde_json::to_string(arg_value).unwrap_or_default(),
                        };
                        invoke_block.push_str("<parameter name=\"");
                        invoke_block.push_str(arg_name);
                        invoke_block.push_str("\">");
                        invoke_block.push_str(&val_str);
                        invoke_block.push_str("</parameter>\n");
                    }
                }
                invoke_block.push_str("</invoke>\n");
            }
            buf.text(&invoke_block, orig_idx)?;
            buf.special(self.tool_call_end, orig_idx);
        }

        buf.special(self.eos, orig_idx);
        buf.text("\n", orig_idx)?;
        Ok(())
    }

    fn emit_tool(
        &self,
        buf: &mut RenderBuf<'_>,
        conversation: &[Message],
        conv_idx: usize,
        orig_idx: i32,
    ) -> Result<(), RenderError> {
        let prev_is_tool = conv_idx > 0 && conversation[conv_idx - 1].role == "tool";
        let next_is_tool =
            conv_idx + 1 < conversation.len() && conversation[conv_idx + 1].role == "tool";

        if !prev_is_tool {
            buf.special(self.role, orig_idx);
            buf.text("tool", orig_idx)?;
        }
        let prefix = if prev_is_tool { "" } else { "\n" };
        let suffix = if next_is_tool { "\n" } else { "" };
        let content = conversation[conv_idx].visible_text_content();
        let mut s = String::with_capacity(content.len() + 32);
        s.push_str(prefix);
        s.push_str("<response>");
        s.push_str(content);
        s.push_str("</response>");
        s.push_str(suffix);
        buf.text(&s, orig_idx)?;

        if !next_is_tool {
            buf.special(self.eos, orig_idx);
            buf.text("\n", orig_idx)?;
        }
        Ok(())
    }
}
