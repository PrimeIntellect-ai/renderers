//! Kimi K2 renderer. Port of `renderers/kimi_k2.py`.
//!
//! Distinctive features:
//!
//! - Per-message framing: `<|im_*|>{role}<|im_middle|>{content}<|im_end|>`.
//!   Role tokens vary by role: `<|im_user|>`, `<|im_assistant|>`,
//!   `<|im_system|>`.
//! - Tool calls wrapped in
//!   `<|tool_calls_section_begin|>` + N × call + `<|tool_calls_section_end|>`,
//!   with each call as
//!   `<|tool_call_begin|>{id}<|tool_call_argument_begin|>{json}<|tool_call_end|>`.
//! - Tool declarations rendered as a `role="tool_declare"` system-style
//!   message with `tojson(separators=(',',':'), sort_keys=True)` JSON.
//! - Tool results: `<|im_system|>{name}<|im_middle|>## Return of {id}\n{content}<|im_end|>`.
//! - Default system message auto-injected if missing
//!   ("You are Kimi, an AI assistant created by Moonshot AI.").
//! - Thinking is plain text `<think>...</think>` (not special tokens).
//!   The template doesn't read `reasoning_content` — assistant content
//!   renders verbatim, inline `<think>` tags and all.

use crate::bridge::{reject_assistant_in_extension, trim_to_turn_close};
use crate::emit::RenderBuf;
use crate::parsing::kimi_k2::parse_kimi_k2;
use crate::tokenizer::Tokenizer;
use crate::traits::Renderer;
use crate::types::{
    Message, ParsedResponse, RenderError, RenderedTokens, SCAFFOLD_IDX, ToolArguments, ToolSpec,
};

const DEFAULT_SYSTEM: &str = "You are Kimi, an AI assistant created by Moonshot AI.";

#[derive(Debug, Clone)]
pub struct KimiK2RendererBuilder {
    enable_thinking: bool,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,
}

impl Default for KimiK2RendererBuilder {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            preserve_all_thinking: false,
            preserve_thinking_between_tool_calls: false,
        }
    }
}

impl KimiK2RendererBuilder {
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
    pub fn build(self, tokenizer: Tokenizer) -> Result<KimiK2Renderer, RenderError> {
        KimiK2Renderer::new_with(tokenizer, &self)
    }
}

#[derive(Debug, Clone)]
pub struct KimiK2Renderer {
    tokenizer: Tokenizer,
    // Stored for API parity; the Kimi template ignores these flags.
    #[allow(dead_code)]
    enable_thinking: bool,
    #[allow(dead_code)]
    preserve_all_thinking: bool,
    #[allow(dead_code)]
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

impl KimiK2Renderer {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, RenderError> {
        KimiK2RendererBuilder::default().build(tokenizer)
    }
    pub fn builder() -> KimiK2RendererBuilder {
        KimiK2RendererBuilder::default()
    }

    fn new_with(tokenizer: Tokenizer, cfg: &KimiK2RendererBuilder) -> Result<Self, RenderError> {
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

    /// Serialise the tools list as compact, key-sorted JSON. The Python
    /// template uses `tojson(separators=(',', ':'), sort_keys=True)` —
    /// match both options here for byte-identical output.
    fn serialize_tools(tools: &[ToolSpec]) -> String {
        // Build an ordered map via serde_json::Map (preserves insertion);
        // for sort_keys behaviour we use a BTreeMap-backed Value tree.
        // serde_json's `serialize` of a BTreeMap sorts keys by Ord<String>.
        use std::collections::BTreeMap;
        let mut arr: Vec<serde_json::Value> = Vec::with_capacity(tools.len());
        for tool in tools {
            let mut m: BTreeMap<String, serde_json::Value> = BTreeMap::new();
            m.insert("name".into(), serde_json::Value::String(tool.name.clone()));
            m.insert(
                "description".into(),
                serde_json::Value::String(tool.description.clone()),
            );
            m.insert("parameters".into(), Self::sort_keys(&tool.parameters));
            if tool.openai_envelope {
                let mut envelope: BTreeMap<String, serde_json::Value> = BTreeMap::new();
                envelope.insert(
                    "function".into(),
                    serde_json::to_value(m).unwrap_or_default(),
                );
                envelope.insert("type".into(), serde_json::Value::String("function".into()));
                arr.push(serde_json::to_value(envelope).unwrap_or_default());
            } else {
                arr.push(serde_json::to_value(m).unwrap_or_default());
            }
        }
        serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
    }

    fn sort_keys(v: &serde_json::Value) -> serde_json::Value {
        use std::collections::BTreeMap;
        match v {
            serde_json::Value::Object(o) => {
                let sorted: BTreeMap<String, serde_json::Value> = o
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::sort_keys(v)))
                    .collect();
                serde_json::to_value(sorted).unwrap_or(serde_json::Value::Null)
            }
            serde_json::Value::Array(a) => {
                serde_json::Value::Array(a.iter().map(Self::sort_keys).collect())
            }
            other => other.clone(),
        }
    }

    fn args_to_string(args: &ToolArguments) -> String {
        match args {
            ToolArguments::Raw(s) => s.clone(),
            ToolArguments::Object(v) => Self::json_dumps_default(v),
        }
    }

    fn json_dumps_default(v: &serde_json::Value) -> String {
        match v {
            serde_json::Value::Null => "null".to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
            serde_json::Value::Array(values) => {
                let inner = values
                    .iter()
                    .map(Self::json_dumps_default)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{inner}]")
            }
            serde_json::Value::Object(values) => {
                let inner = values
                    .iter()
                    .map(|(key, value)| {
                        let key = serde_json::to_string(key).unwrap_or_default();
                        let value = Self::json_dumps_default(value);
                        format!("{key}: {value}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{{inner}}}")
            }
        }
    }

    fn emit_im_role(
        &self,
        buf: &mut RenderBuf<'_>,
        role_token: u32,
        role_name: &str,
        content: &str,
        idx: i32,
    ) -> Result<(), RenderError> {
        buf.special(role_token, idx);
        buf.text(role_name, idx)?;
        buf.special(self.im_middle, idx);
        buf.text(content, idx)?;
        buf.special(self.im_end, idx);
        Ok(())
    }
}

impl Renderer for KimiK2Renderer {
    fn render(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError> {
        if messages.is_empty() {
            return Err(RenderError::EmptyMessages);
        }

        // Inject tool_declare + default system into a working copy, tracking
        // which slots are injected so message_indices stays aligned to the
        // caller's original list.
        let mut working: Vec<Message> = Vec::with_capacity(messages.len() + 2);
        let mut injected: Vec<bool> = Vec::with_capacity(messages.len() + 2);

        // tool_declare goes first if tools were provided and the caller
        // didn't already include a tool_declare message.
        let tools_pending = tools.is_some_and(|t| !t.is_empty());
        let already_has_tool_declare = !messages.is_empty() && messages[0].role == "tool_declare";
        if tools_pending && !already_has_tool_declare {
            working.push(Message {
                role: "tool_declare".to_string(),
                content: crate::types::Content::Text(Self::serialize_tools(tools.unwrap())),
                ..Default::default()
            });
            injected.push(true);
        }

        // Then the optional default system message
        let auto_system_position: Option<usize> =
            if !messages.is_empty() && messages[0].role == "tool_declare" {
                // tool_declare present in caller's input → if next isn't system,
                // inject default system AFTER tool_declare
                if messages.len() < 2 || messages[1].role != "system" {
                    Some(working.len() + 1) // will be inserted between tool_declare and the rest
                } else {
                    None
                }
            } else if messages.is_empty() || messages[0].role != "system" {
                Some(working.len())
            } else {
                None
            };

        // Now lay out the rest:
        if let Some(pos) = auto_system_position {
            // Replicate the Python logic: if caller's first message is
            // tool_declare, push it then the default system then the rest.
            if !messages.is_empty() && messages[0].role == "tool_declare" {
                working.push(messages[0].clone());
                injected.push(false);
                working.push(Message {
                    role: "system".to_string(),
                    content: crate::types::Content::Text(DEFAULT_SYSTEM.to_string()),
                    ..Default::default()
                });
                injected.push(true);
                for m in &messages[1..] {
                    working.push(m.clone());
                    injected.push(false);
                }
            } else {
                working.push(Message {
                    role: "system".to_string(),
                    content: crate::types::Content::Text(DEFAULT_SYSTEM.to_string()),
                    ..Default::default()
                });
                injected.push(true);
                for m in messages {
                    working.push(m.clone());
                    injected.push(false);
                }
            }
            let _ = pos;
        } else {
            for m in messages {
                working.push(m.clone());
                injected.push(false);
            }
        }

        // Map normalised index → caller's index (sentinel for injected).
        // Precompute as a flat Vec so the lookup is O(1) instead of an
        // O(i) filter inside the render loop — saves an O(n²) walk on
        // long conversations.
        let orig_idx_table: Vec<i32> = {
            let mut table = Vec::with_capacity(working.len());
            let mut real: i32 = -1;
            for &inj in &injected {
                if inj {
                    table.push(SCAFFOLD_IDX);
                } else {
                    real += 1;
                    table.push(real);
                }
            }
            table
        };
        let orig_idx = |i: usize| -> i32 { orig_idx_table[i] };

        // Index of the auto-injected system message (if any) — emits a
        // trailing literal "\n" after its <|im_end|>.
        let auto_system_idx: Option<usize> = working
            .iter()
            .enumerate()
            .find(|(i, m)| injected[*i] && m.role == "system")
            .map(|(i, _)| i);

        let mut buf = RenderBuf::new(
            &self.tokenizer,
            working.len().max(1) * 256 + tools.map_or(0, |t| 64 * t.len() + 256),
        );

        for (i, msg) in working.iter().enumerate() {
            let oi = orig_idx(i);
            let content = msg.text_content();
            match msg.role.as_str() {
                "system" => {
                    self.emit_im_role(&mut buf, self.im_system, "system", content, oi)?;
                    if Some(i) == auto_system_idx {
                        buf.text("\n", oi)?;
                    }
                }
                "tool_declare" => {
                    self.emit_im_role(&mut buf, self.im_system, "tool_declare", content, oi)?;
                }
                "user" => {
                    self.emit_im_role(&mut buf, self.im_user, "user", content, oi)?;
                }
                "assistant" => self.emit_assistant(&mut buf, msg, oi)?,
                "tool" => self.emit_tool(&mut buf, msg, content, oi)?,
                other => {
                    // Unknown role: render system-style
                    self.emit_im_role(&mut buf, self.im_system, other, content, oi)?;
                }
            }
        }

        if add_generation_prompt {
            buf.scaffold_special(self.im_assistant);
            buf.scaffold_text("assistant")?;
            buf.scaffold_special(self.im_middle);
        }

        Ok(buf.into_rendered())
    }

    fn parse_response(&self, token_ids: &[u32]) -> ParsedResponse {
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
            let content = msg.text_content();
            match msg.role.as_str() {
                "user" => self.emit_im_role(&mut buf, self.im_user, "user", content, idx)?,
                "system" => self.emit_im_role(&mut buf, self.im_system, "system", content, idx)?,
                "tool" => self.emit_tool(&mut buf, msg, content, idx)?,
                _ => return Ok(None),
            }
        }

        buf.scaffold_special(self.im_assistant);
        buf.scaffold_text("assistant")?;
        buf.scaffold_special(self.im_middle);

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

impl KimiK2Renderer {
    fn emit_assistant(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        idx: i32,
    ) -> Result<(), RenderError> {
        buf.special(self.im_assistant, idx);
        buf.text("assistant", idx)?;
        buf.special(self.im_middle, idx);

        // Kimi's template renders content verbatim; reasoning_content is
        // ignored (not read by the Jinja).
        buf.text(msg.text_content(), idx)?;

        if !msg.tool_calls.is_empty() {
            buf.special(self.tool_calls_section_begin, idx);
            for tc in &msg.tool_calls {
                let args_str = Self::args_to_string(&tc.function.arguments);
                let tc_id = tc.id.clone().unwrap_or_default();
                buf.special(self.tool_call_begin, idx);
                buf.text(&tc_id, idx)?;
                buf.special(self.tool_call_argument_begin, idx);
                buf.text(&args_str, idx)?;
                buf.special(self.tool_call_end, idx);
            }
            buf.special(self.tool_calls_section_end, idx);
        }
        buf.special(self.im_end, idx);
        Ok(())
    }

    fn emit_tool(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        content: &str,
        idx: i32,
    ) -> Result<(), RenderError> {
        let name = msg.name.as_deref().unwrap_or("tool");
        let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");
        buf.special(self.im_system, idx);
        buf.text(name, idx)?;
        buf.special(self.im_middle, idx);
        let mut header = String::with_capacity(tool_call_id.len() + 16);
        header.push_str("## Return of ");
        header.push_str(tool_call_id);
        header.push('\n');
        buf.text(&header, idx)?;
        buf.text(content, idx)?;
        buf.special(self.im_end, idx);
        Ok(())
    }
}
