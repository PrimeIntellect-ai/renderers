//! Nemotron 3 renderer. Port of `renderers/nemotron3.py`.
//!
//! Same `<|im_start|>/<|im_end|>` framing as Qwen3.5, but with several
//! template-specific quirks:
//!
//! - Tool declarations use XML (`<function>...</function>` with nested
//!   `<parameter>` blocks), not JSON-per-line.
//! - System prompt is emitted BEFORE the tools block (Qwen3.5 puts
//!   tools first).
//! - An empty system message is auto-injected if none is present.
//! - `<think></think>` is emitted on EVERY assistant message, even
//!   those without reasoning content (collapses to empty block).
//! - Single `\n` after `</think>` (Qwen3.5 uses `\n\n`).
//! - Disable-thinking generation suffix is `<think></think>` with no
//!   trailing newlines.
//! - Trailing `\n` after `</tool_response>`.
//! - `<|endoftext|>` is *optional* — Nemotron-3 Nano / Super ship with
//!   only `<|im_end|>` as EOS; larger variants additionally include
//!   `<|endoftext|>`.

use serde_json::Value as JsonValue;

use crate::bridge::{reject_assistant_in_extension, trim_to_turn_close};
use crate::emit::RenderBuf;
use crate::parsing::qwen35::parse_qwen35;
use crate::thinking::should_preserve_past_thinking;
use crate::tokenizer::Tokenizer;
use crate::tool_cache::ToolTextCache;
use crate::traits::Renderer;
use crate::types::{
    Message, ParsedResponse, RenderError, RenderedTokens, SCAFFOLD_IDX, ToolArguments, ToolSpec,
};

const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_FOOTER: &str = "\n</tools>";
const TOOLS_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

#[derive(Debug, Clone)]
pub struct Nemotron3RendererBuilder {
    enable_thinking: bool,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,
}

impl Default for Nemotron3RendererBuilder {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            preserve_all_thinking: false,
            preserve_thinking_between_tool_calls: false,
        }
    }
}

impl Nemotron3RendererBuilder {
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
    pub fn build(self, tokenizer: Tokenizer) -> Result<Nemotron3Renderer, RenderError> {
        Nemotron3Renderer::new_with(tokenizer, &self)
    }
}

#[derive(Debug, Clone)]
pub struct Nemotron3Renderer {
    tokenizer: Tokenizer,
    enable_thinking: bool,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,

    im_start: u32,
    im_end: u32,
    /// `<|endoftext|>` is optional — Nemotron-3 Nano / Super tokenizers
    /// don't ship it.
    endoftext: Option<u32>,
    think: u32,
    think_end: u32,
    tool_call: u32,
    tool_call_end: u32,
    tool_response: u32,
    tool_response_end: u32,

    stop_tokens: Vec<u32>,
    newline_tokens: Vec<u32>,
    system_newline_tokens: Vec<u32>,
    user_newline_tokens: Vec<u32>,
    assistant_newline_tokens: Vec<u32>,
    function_close_newline_tokens: Vec<u32>,
    tool_text_cache: ToolTextCache,
}

impl Nemotron3Renderer {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, RenderError> {
        Nemotron3RendererBuilder::default().build(tokenizer)
    }
    pub fn builder() -> Nemotron3RendererBuilder {
        Nemotron3RendererBuilder::default()
    }

    fn new_with(tokenizer: Tokenizer, cfg: &Nemotron3RendererBuilder) -> Result<Self, RenderError> {
        let im_start = tokenizer.token_to_id_strict("<|im_start|>")?;
        let im_end = tokenizer.token_to_id_strict("<|im_end|>")?;
        let endoftext = tokenizer.token_to_id("<|endoftext|>");
        let think = tokenizer.token_to_id_strict("<think>")?;
        let think_end = tokenizer.token_to_id_strict("</think>")?;
        let tool_call = tokenizer.token_to_id_strict("<tool_call>")?;
        let tool_call_end = tokenizer.token_to_id_strict("</tool_call>")?;
        let tool_response = tokenizer.token_to_id_strict("<tool_response>")?;
        let tool_response_end = tokenizer.token_to_id_strict("</tool_response>")?;

        let mut stop_tokens = vec![im_end];
        if let Some(eot) = endoftext {
            stop_tokens.push(eot);
        }
        let newline_tokens = tokenizer.encode_no_special("\n")?.as_slice().to_vec();
        let system_newline_tokens = tokenizer.encode_no_special("system\n")?.as_slice().to_vec();
        let user_newline_tokens = tokenizer.encode_no_special("user\n")?.as_slice().to_vec();
        let assistant_newline_tokens = tokenizer
            .encode_no_special("assistant\n")?
            .as_slice()
            .to_vec();
        let function_close_newline_tokens = tokenizer
            .encode_no_special("</function>\n")?
            .as_slice()
            .to_vec();

        Ok(Self {
            tokenizer,
            enable_thinking: cfg.enable_thinking,
            preserve_all_thinking: cfg.preserve_all_thinking,
            preserve_thinking_between_tool_calls: cfg.preserve_thinking_between_tool_calls,
            im_start,
            im_end,
            endoftext,
            think,
            think_end,
            tool_call,
            tool_call_end,
            tool_response,
            tool_response_end,
            stop_tokens,
            newline_tokens,
            system_newline_tokens,
            user_newline_tokens,
            assistant_newline_tokens,
            function_close_newline_tokens,
            tool_text_cache: ToolTextCache::default(),
        })
    }

    /// Render a single tool declaration in Nemotron 3's XML format.
    /// Mirrors `_format_tool_declaration` in the Python impl.
    fn format_tool_declaration(tool: &ToolSpec) -> String {
        let mut out = String::with_capacity(256);
        out.push_str("<function>\n<name>");
        out.push_str(&tool.name);
        out.push_str("</name>");
        let desc = tool.description.trim();
        if !desc.is_empty() {
            out.push_str("\n<description>");
            out.push_str(desc);
            out.push_str("</description>");
        }
        out.push_str("\n<parameters>");

        if let Some(props) = tool
            .parameters
            .get("properties")
            .and_then(|v| v.as_object())
        {
            for (param_name, param_fields) in props {
                out.push_str("\n<parameter>\n<name>");
                out.push_str(param_name);
                out.push_str("</name>");
                if let Some(t) = param_fields.get("type") {
                    out.push_str("\n<type>");
                    Self::write_value_as_text(&mut out, t);
                    out.push_str("</type>");
                }
                if let Some(d) = param_fields.get("description").and_then(|v| v.as_str()) {
                    out.push_str("\n<description>");
                    out.push_str(d.trim());
                    out.push_str("</description>");
                }
                if let Some(e) = param_fields.get("enum") {
                    out.push_str("\n<enum>");
                    out.push_str(&serde_json::to_string(e).unwrap_or_default());
                    out.push_str("</enum>");
                }
                if let Some(obj) = param_fields.as_object() {
                    Self::render_extra_keys(
                        &mut out,
                        obj,
                        &["name", "type", "description", "enum"],
                    );
                }
                out.push_str("\n</parameter>");
            }
        }
        if let Some(obj) = tool.parameters.as_object() {
            Self::render_extra_keys(&mut out, obj, &["type", "properties", "required"]);
        }
        if let Some(req) = tool.parameters.get("required") {
            out.push_str("\n<required>");
            out.push_str(&serde_json::to_string(req).unwrap_or_default());
            out.push_str("</required>");
        }
        out.push_str("\n</parameters>");
        out.push_str("\n</function>");
        out
    }

    /// Mirror Python's `str(value)` for non-string JSON values
    /// (used inside `<type>{value}</type>` tags).
    fn write_value_as_text(out: &mut String, value: &JsonValue) {
        match value {
            JsonValue::String(s) => out.push_str(s),
            JsonValue::Bool(true) => out.push_str("True"),
            JsonValue::Bool(false) => out.push_str("False"),
            JsonValue::Null => out.push_str("None"),
            JsonValue::Number(n) => out.push_str(&n.to_string()),
            _ => out.push_str(&serde_json::to_string(value).unwrap_or_default()),
        }
    }

    /// Mirror Python's `_render_extra_keys` — emit `<key>value</key>`
    /// for every key not already handled.
    fn render_extra_keys(
        out: &mut String,
        obj: &serde_json::Map<String, JsonValue>,
        handled: &[&str],
    ) {
        for (k, v) in obj {
            if handled.contains(&k.as_str()) {
                continue;
            }
            out.push_str("\n<");
            out.push_str(k);
            out.push('>');
            match v {
                JsonValue::Object(_) | JsonValue::Array(_) => {
                    out.push_str(&serde_json::to_string(v).unwrap_or_default());
                }
                _ => Self::write_value_as_text(out, v),
            }
            out.push_str("</");
            out.push_str(k);
            out.push('>');
        }
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
        buf.ids(&self.system_newline_tokens, sys_idx);

        let system_content = if first_is_system {
            messages[0].text_content().trim().to_string()
        } else {
            String::new()
        };
        let tool_tokens = self.tool_text_cache.get_or_insert_with(
            &self.tokenizer,
            tools,
            u64::from(first_is_system),
            &system_content,
            || {
                let mut full_sys = String::with_capacity(512);
                full_sys.push_str(&system_content);
                let mut tools_block = String::with_capacity(512);
                tools_block.push_str(TOOLS_HEADER);
                tools_block.push('\n');
                let mut first = true;
                for t in tools {
                    if !first {
                        tools_block.push('\n');
                    }
                    tools_block.push_str(&Self::format_tool_declaration(t));
                    first = false;
                }
                tools_block.push_str(TOOLS_FOOTER);
                tools_block.push_str(TOOLS_INSTRUCTIONS);

                if !full_sys.is_empty() {
                    full_sys.push_str("\n\n");
                }
                full_sys.push_str(&tools_block);
                Ok(full_sys)
            },
        )?;
        buf.ids(tool_tokens.as_slice(), sys_idx);
        buf.special(self.im_end, sys_idx);
        buf.ids(&self.newline_tokens, sys_idx);
        Ok(())
    }

    fn emit_system_no_tools(
        &self,
        buf: &mut RenderBuf<'_>,
        messages: &[Message],
        sys_idx: i32,
    ) -> Result<(), RenderError> {
        let content = messages[0].text_content().trim();
        buf.special(self.im_start, sys_idx);
        let mut s = String::with_capacity(content.len() + 8);
        s.push_str("system\n");
        s.push_str(content);
        buf.text(&s, sys_idx)?;
        buf.special(self.im_end, sys_idx);
        buf.ids(&self.newline_tokens, sys_idx);
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
        buf.ids(&self.newline_tokens, idx);
        Ok(())
    }

    fn emit_tool(
        &self,
        buf: &mut RenderBuf<'_>,
        messages: &[Message],
        msg_idx: usize,
        content: &str,
        msg_orig_idx: i32,
    ) -> Result<(), RenderError> {
        let prev_is_tool = msg_idx > 0 && messages[msg_idx - 1].role == "tool";
        let next_is_tool = msg_idx + 1 < messages.len() && messages[msg_idx + 1].role == "tool";

        if !prev_is_tool {
            buf.special(self.im_start, msg_orig_idx);
            buf.ids(&self.user_newline_tokens, msg_orig_idx);
        }
        buf.special(self.tool_response, msg_orig_idx);
        let mut wrapped = String::with_capacity(content.len() + 2);
        wrapped.push('\n');
        wrapped.push_str(content);
        wrapped.push('\n');
        buf.text(&wrapped, msg_orig_idx)?;
        buf.special(self.tool_response_end, msg_orig_idx);
        // Nemotron 3: trailing \n after </tool_response>
        buf.ids(&self.newline_tokens, msg_orig_idx);

        if !next_is_tool {
            buf.special(self.im_end, msg_orig_idx);
            buf.ids(&self.newline_tokens, msg_orig_idx);
        }
        Ok(())
    }

    fn emit_assistant(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        msg_orig_idx: i32,
        is_last_turn: bool,
        preserve_thinking: bool,
    ) -> Result<(), RenderError> {
        // Recover reasoning_content either from the field or from inline tags.
        let raw_content = msg.text_content().trim();
        let (reasoning_content, content) = match &msg.reasoning_content {
            Some(s) => (s.clone(), raw_content.to_string()),
            None => {
                if let Some((before, after)) = raw_content.split_once("</think>") {
                    let r = if let Some((_, inner)) = before.rsplit_once("<think>") {
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
                    (r, after.trim_start_matches('\n').to_string())
                } else {
                    (String::new(), raw_content.to_string())
                }
            }
        };
        let reasoning_content = reasoning_content.trim().to_string();

        buf.special(self.im_start, msg_orig_idx);
        buf.ids(&self.assistant_newline_tokens, msg_orig_idx);

        let tool_calls = &msg.tool_calls;
        let content_suffix = if tool_calls.is_empty() { "" } else { "\n" };

        if !reasoning_content.is_empty() && (is_last_turn || preserve_thinking) {
            buf.special(self.think, msg_orig_idx);
            let mut s = String::with_capacity(reasoning_content.len() + 2);
            s.push('\n');
            s.push_str(&reasoning_content);
            s.push('\n');
            buf.text(&s, msg_orig_idx)?;
            buf.special(self.think_end, msg_orig_idx);
            // Single \n separator (not \n\n like Qwen3.5)
            let mut tail = String::with_capacity(content.len() + 2);
            tail.push('\n');
            tail.push_str(&content);
            tail.push_str(content_suffix);
            buf.text(&tail, msg_orig_idx)?;
        } else if !reasoning_content.is_empty() {
            // Historical assistant whose reasoning got stripped — collapsed
            // <think></think> + single \n + content.
            buf.special(self.think, msg_orig_idx);
            buf.special(self.think_end, msg_orig_idx);
            let mut tail = String::with_capacity(content.len() + 2);
            tail.push('\n');
            tail.push_str(&content);
            tail.push_str(content_suffix);
            buf.text(&tail, msg_orig_idx)?;
        } else {
            // No reasoning ever — <think></think> glued directly to content.
            buf.special(self.think, msg_orig_idx);
            buf.special(self.think_end, msg_orig_idx);
            let mut tail = String::with_capacity(content.len() + 2);
            tail.push_str(&content);
            tail.push_str(content_suffix);
            buf.text(&tail, msg_orig_idx)?;
        }

        for tc in tool_calls {
            let name = tc.function.name.as_str();
            buf.special(self.tool_call, msg_orig_idx);
            let mut head = String::with_capacity(name.len() + 16);
            head.push_str("\n<function=");
            head.push_str(name);
            head.push_str(">\n");
            buf.text(&head, msg_orig_idx)?;

            let args_value = match &tc.function.arguments {
                ToolArguments::Object(v) => v.clone(),
                ToolArguments::Raw(s) => {
                    serde_json::from_str(s).unwrap_or(JsonValue::Object(serde_json::Map::new()))
                }
            };
            if let Some(obj) = args_value.as_object() {
                for (arg_name, arg_value) in obj {
                    let val_str = match arg_value {
                        JsonValue::Object(_) | JsonValue::Array(_) => {
                            serde_json::to_string(arg_value).unwrap_or_default()
                        }
                        JsonValue::String(s) => s.clone(),
                        JsonValue::Bool(b) => {
                            if *b {
                                "True".into()
                            } else {
                                "False".into()
                            }
                        }
                        JsonValue::Null => "None".into(),
                        JsonValue::Number(n) => n.to_string(),
                    };
                    let mut param = String::with_capacity(arg_name.len() + val_str.len() + 24);
                    param.push_str("<parameter=");
                    param.push_str(arg_name);
                    param.push_str(">\n");
                    param.push_str(&val_str);
                    param.push_str("\n</parameter>\n");
                    buf.text(&param, msg_orig_idx)?;
                }
            }

            buf.ids(&self.function_close_newline_tokens, msg_orig_idx);
            buf.special(self.tool_call_end, msg_orig_idx);
            // Nemotron 3: trailing \n after </tool_call>
            buf.ids(&self.newline_tokens, msg_orig_idx);
        }

        buf.special(self.im_end, msg_orig_idx);
        buf.ids(&self.newline_tokens, msg_orig_idx);
        Ok(())
    }

    fn emit_generation_prompt(&self, buf: &mut RenderBuf<'_>) {
        buf.scaffold_special(self.im_start);
        buf.ids(&self.assistant_newline_tokens, SCAFFOLD_IDX);
        if self.enable_thinking {
            buf.scaffold_special(self.think);
            buf.ids(&self.newline_tokens, SCAFFOLD_IDX);
        } else {
            // Disable-thinking suffix: <think></think> with no trailing newlines
            buf.scaffold_special(self.think);
            buf.scaffold_special(self.think_end);
        }
    }

    fn estimate_capacity(messages: &[Message], tools: Option<&[ToolSpec]>) -> usize {
        let base = messages.len().max(1) * 256;
        let tools_bonus = tools.map_or(0, |t| 384 * t.len().max(1) + 512);
        base + tools_bonus
    }
}

impl Renderer for Nemotron3Renderer {
    fn render(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError> {
        if messages.is_empty() {
            return Err(RenderError::EmptyMessages);
        }
        let mut buf = RenderBuf::new(&self.tokenizer, Self::estimate_capacity(messages, tools));

        // Normalise: prepend empty system message if none is present.
        let mut normalised: Vec<Message>;
        let auto_system_injected: bool;
        let messages_ref: &[Message] = if messages[0].role == "system" {
            auto_system_injected = false;
            messages
        } else {
            auto_system_injected = true;
            normalised = Vec::with_capacity(messages.len() + 1);
            normalised.push(Message {
                role: "system".to_string(),
                content: crate::types::Content::Text(String::new()),
                ..Default::default()
            });
            normalised.extend_from_slice(messages);
            &normalised
        };

        // Map normalised index back to caller's original index. Injected
        // system uses SCAFFOLD_IDX (-1) so build_training_sample can't
        // dereference past the caller's input.
        let orig_idx = |i: usize| -> i32 {
            if auto_system_injected {
                if i == 0 { SCAFFOLD_IDX } else { (i - 1) as i32 }
            } else {
                i as i32
            }
        };

        let first_is_system = messages_ref[0].role == "system";

        match tools {
            Some(t) if !t.is_empty() => {
                self.emit_system_with_tools(&mut buf, messages_ref, t, first_is_system)?;
            }
            _ => {
                if first_is_system {
                    self.emit_system_no_tools(&mut buf, messages_ref, orig_idx(0))?;
                }
            }
        }

        // Find the most-recent plain (non-tool-call) assistant — reasoning
        // is preserved on it and on later turns; earlier assistants
        // collapse to <think></think>.
        let last_plain_assistant_idx: i32 = {
            let mut found: i32 = -1;
            for (j, m) in messages_ref.iter().enumerate().rev() {
                if m.role == "assistant" && m.tool_calls.is_empty() {
                    found = j as i32;
                    break;
                }
            }
            found
        };

        for (i, msg) in messages_ref.iter().enumerate() {
            let content = msg.text_content().trim();
            let oi = orig_idx(i);
            match msg.role.as_str() {
                "system" => {
                    if i != 0 {
                        return Err(RenderError::Invalid(
                            "system message must be at the beginning".into(),
                        ));
                    }
                    // Already handled above
                }
                "user" => self.emit_user(&mut buf, content, oi)?,
                "assistant" => {
                    let is_last_turn = (i as i32) >= last_plain_assistant_idx;
                    // oi >= 0 guard above makes the usize cast safe.
                    #[allow(clippy::cast_sign_loss)]
                    let preserve_thinking = oi >= 0
                        && should_preserve_past_thinking(
                            messages,
                            oi as usize,
                            self.preserve_all_thinking,
                            self.preserve_thinking_between_tool_calls,
                        );
                    self.emit_assistant(&mut buf, msg, oi, is_last_turn, preserve_thinking)?;
                }
                "tool" => self.emit_tool(&mut buf, messages_ref, i, content, oi)?,
                _ => {
                    return Err(RenderError::Invalid(format!(
                        "unexpected message role: {}",
                        msg.role
                    )));
                }
            }
        }

        if add_generation_prompt {
            self.emit_generation_prompt(&mut buf);
        }

        Ok(buf.into_rendered())
    }

    fn parse_response(&self, token_ids: &[u32]) -> ParsedResponse {
        parse_qwen35(
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
            Some(self.im_end),
        ) else {
            return Ok(None);
        };

        let mut buf = RenderBuf::new_token_ids_only(
            &self.tokenizer,
            Self::estimate_capacity(new_messages, None),
        );
        buf.ids(&self.newline_tokens, SCAFFOLD_IDX);

        for (i, msg) in new_messages.iter().enumerate() {
            let content = msg.text_content().trim();
            let idx = i as i32;
            match msg.role.as_str() {
                "user" => self.emit_user(&mut buf, content, idx)?,
                "system" => {
                    buf.special(self.im_start, idx);
                    let mut s = String::with_capacity(content.len() + 8);
                    s.push_str("system\n");
                    s.push_str(content);
                    buf.text(&s, idx)?;
                    buf.special(self.im_end, idx);
                    buf.ids(&self.newline_tokens, idx);
                }
                "tool" => self.emit_tool(&mut buf, new_messages, i, content, idx)?,
                _ => return Ok(None),
            }
        }

        self.emit_generation_prompt(&mut buf);

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

// Keep the field readable; suppresses dead-code warning since we only use it via the Option arm above.
#[allow(dead_code)]
impl Nemotron3Renderer {
    pub fn has_endoftext(&self) -> bool {
        self.endoftext.is_some()
    }
}
