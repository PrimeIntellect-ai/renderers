//! Qwen3.5 renderer (text-only). Port of `renderers/qwen35.py` minus the
//! multimodal path; multimodal lands in Phase 5 with the vision processor.
//!
//! Differences from Qwen3:
//!
//! - `<think>` / `</think>` are **special tokens**, not text tags.
//! - Tool calls use XML format with `<function=name>` and
//!   `<parameter=key>` blocks.
//! - System prompt includes a verbose tool-instructions block.
//! - Generation prompt prefills `<think>\n` (or the empty-think block
//!   when `enable_thinking` is false), with polarity defaulting to
//!   `enable_thinking=true` for big-size models.
//!
//! `enable_thinking` polarity detection (which the Python implementation
//! probes via a one-shot `apply_chat_template` call) is **not** done in
//! Rust — the caller passes it explicitly through the builder. The
//! Python shim handles the polarity probe and forwards the result.

use std::borrow::Cow;

use crate::bridge::{reject_assistant_in_extension, trim_to_turn_close};
use crate::emit::RenderBuf;
use crate::json::{to_string_python, tool_spec_template_value};
use crate::parsing::qwen35::parse_qwen35;
use crate::thinking::should_preserve_past_thinking;
use crate::tokenizer::Tokenizer;
use crate::traits::{MultimodalRenderer, Renderer};
use crate::types::{
    Content, ContentPart, MediaBundle, MediaItem, Message, Modality, MultiModalData,
    ParsedResponse, PlaceholderRange, RenderError, RenderedTokens, SCAFFOLD_IDX, ToolArguments,
    ToolSpec,
};

const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_FOOTER: &str = "\n</tools>";
const TOOLS_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

#[derive(Debug, Clone)]
pub struct Qwen35RendererBuilder {
    enable_thinking: bool,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,
    /// When `true`, every non-string tool-call argument is serialised via
    /// `serde_json::to_string` instead of Python's `str(...)` rules. This
    /// is the only behavioural change Qwen3.6 introduces vs Qwen3.5 —
    /// kept as a flag here so Qwen3.6 is a config delta, not a code
    /// duplicate.
    args_as_json: bool,
}

impl Default for Qwen35RendererBuilder {
    fn default() -> Self {
        Self {
            // Big-size model default. The Python shim probes the tokenizer's
            // Jinja template to discover the per-model polarity; callers can
            // pass an explicit override here.
            enable_thinking: true,
            preserve_all_thinking: false,
            preserve_thinking_between_tool_calls: false,
            args_as_json: false,
        }
    }
}

impl Qwen35RendererBuilder {
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
    /// Qwen3.6 flag — JSON-serialise every non-string tool argument.
    pub fn args_as_json(mut self, on: bool) -> Self {
        self.args_as_json = on;
        self
    }
    pub fn build(self, tokenizer: Tokenizer) -> Result<Qwen35Renderer, RenderError> {
        Qwen35Renderer::new_with(tokenizer, self)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen35Renderer {
    tokenizer: Tokenizer,
    enable_thinking: bool,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,
    args_as_json: bool,

    im_start: u32,
    im_end: u32,
    #[allow(dead_code)]
    endoftext: u32,
    think: u32,
    think_end: u32,
    tool_call: u32,
    tool_call_end: u32,
    tool_response: u32,
    tool_response_end: u32,

    // Multimodal placeholder tokens — resolved as optional so the
    // text-only Qwen3.5 tokenizers (which don't ship the vision
    // specials) still construct cleanly. `as_multimodal()` returns None
    // when these are absent.
    vision_start: Option<u32>,
    vision_end: Option<u32>,
    image_pad: Option<u32>,
    video_pad: Option<u32>,
    /// `[(token_id, modality_marker)]` — 1 = image, 2 = video. Empty
    /// when this tokenizer doesn't have the vision specials.
    mm_token_type_ids: Vec<(u32, u8)>,

    stop_tokens: Vec<u32>,
}

impl Qwen35Renderer {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, RenderError> {
        Qwen35RendererBuilder::default().build(tokenizer)
    }

    pub fn builder() -> Qwen35RendererBuilder {
        Qwen35RendererBuilder::default()
    }

    fn new_with(tokenizer: Tokenizer, cfg: Qwen35RendererBuilder) -> Result<Self, RenderError> {
        let im_start = tokenizer.token_to_id_strict("<|im_start|>")?;
        let im_end = tokenizer.token_to_id_strict("<|im_end|>")?;
        let endoftext = tokenizer.token_to_id_strict("<|endoftext|>")?;
        let think = tokenizer.token_to_id_strict("<think>")?;
        let think_end = tokenizer.token_to_id_strict("</think>")?;
        let tool_call = tokenizer.token_to_id_strict("<tool_call>")?;
        let tool_call_end = tokenizer.token_to_id_strict("</tool_call>")?;
        let tool_response = tokenizer.token_to_id_strict("<tool_response>")?;
        let tool_response_end = tokenizer.token_to_id_strict("</tool_response>")?;

        // Multimodal tokens are optional — text-only tokenizers (e.g.
        // Qwen3.5-9B, no `-VL` suffix) don't ship them. Resolve via
        // `token_to_id` (non-strict) so the renderer constructs in both
        // cases.
        let vision_start = tokenizer.token_to_id("<|vision_start|>");
        let vision_end = tokenizer.token_to_id("<|vision_end|>");
        let image_pad = tokenizer.token_to_id("<|image_pad|>");
        let video_pad = tokenizer.token_to_id("<|video_pad|>");
        let mut mm_token_type_ids: Vec<(u32, u8)> = Vec::new();
        if let Some(p) = image_pad {
            mm_token_type_ids.push((p, 1));
        }
        if let Some(p) = video_pad {
            mm_token_type_ids.push((p, 2));
        }

        Ok(Self {
            tokenizer,
            enable_thinking: cfg.enable_thinking,
            preserve_all_thinking: cfg.preserve_all_thinking,
            preserve_thinking_between_tool_calls: cfg.preserve_thinking_between_tool_calls,
            args_as_json: cfg.args_as_json,
            im_start,
            im_end,
            endoftext,
            think,
            think_end,
            tool_call,
            tool_call_end,
            tool_response,
            tool_response_end,
            vision_start,
            vision_end,
            image_pad,
            video_pad,
            mm_token_type_ids,
            stop_tokens: vec![im_end, endoftext],
        })
    }

    /// True when the underlying tokenizer ships the vision special
    /// tokens. Used by [`Renderer::as_multimodal`].
    pub fn supports_multimodal(&self) -> bool {
        self.vision_start.is_some() && self.vision_end.is_some() && self.image_pad.is_some()
    }

    /// Text view of message content, matching the Python
    /// `Qwen35Renderer._render_content` helper: join text parts and skip
    /// media / thinking parts. This is used by the text-only native path
    /// so OpenAI-style structured text content is not silently dropped.
    fn render_content_text(content: &Content) -> Cow<'_, str> {
        match content {
            Content::Null => Cow::Borrowed(""),
            Content::Text(s) => Cow::Borrowed(s.as_str()),
            Content::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        out.push_str(text);
                    }
                }
                Cow::Owned(out)
            }
        }
    }

    /// Index of the most recent non-tool-response user message;
    /// `messages.len()` when none — that out-of-range value makes
    /// `msg_idx > last_query_index` uniformly false, matching the
    /// Python contract.
    fn last_query_index(messages: &[Message]) -> i32 {
        for (i, msg) in messages.iter().enumerate().rev() {
            if msg.role != "user" {
                continue;
            }
            let content = Self::render_content_text(&msg.content);
            let content = content.trim();
            if !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>")) {
                return i as i32;
            }
        }
        messages.len() as i32
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
        buf.text("system\n", sys_idx)?;

        let mut tool_text =
            String::with_capacity(TOOLS_HEADER.len() + TOOLS_INSTRUCTIONS.len() + 256);
        tool_text.push_str(TOOLS_HEADER);
        for tool in tools {
            tool_text.push('\n');
            let spec = tool_spec_template_value(tool);
            tool_text.push_str(&to_string_python(&spec).map_err(|e| {
                RenderError::Invalid(format!("tool spec serialisation failed: {e}"))
            })?);
        }
        tool_text.push_str(TOOLS_FOOTER);
        tool_text.push_str(TOOLS_INSTRUCTIONS);

        if first_is_system {
            let sys_content = Self::render_content_text(&messages[0].content);
            let sys_content = sys_content.trim();
            if !sys_content.is_empty() {
                tool_text.push_str("\n\n");
                tool_text.push_str(sys_content);
            }
        }

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
        let content = Self::render_content_text(&messages[0].content);
        let content = content.trim();
        buf.special(self.im_start, 0);
        let mut s = String::with_capacity(content.len() + 8);
        s.push_str("system\n");
        s.push_str(content);
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

    fn render_arg_value(arg_value: &serde_json::Value, args_as_json: bool) -> String {
        if args_as_json {
            // Qwen3.6: every non-string serialises via serde_json (bools
            // become "true"/"false", None becomes "null"). Strings still
            // render verbatim — JSON would re-quote them.
            match arg_value {
                serde_json::Value::String(s) => s.clone(),
                _ => serde_json::to_string(arg_value).unwrap_or_default(),
            }
        } else {
            // Qwen3.5: Python's str() rules — dict/list go through JSON,
            // bools become "True"/"False", None becomes "None", numbers
            // and strings render verbatim.
            match arg_value {
                serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                    serde_json::to_string(arg_value).unwrap_or_default()
                }
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Bool(b) => {
                    if *b {
                        "True".to_string()
                    } else {
                        "False".to_string()
                    }
                }
                serde_json::Value::Null => "None".to_string(),
                serde_json::Value::Number(n) => n.to_string(),
            }
        }
    }

    fn emit_assistant(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        msg_idx: usize,
        last_query_index: i32,
        preserve_thinking: bool,
    ) -> Result<(), RenderError> {
        let raw_content = Self::render_content_text(&msg.content);
        let (reasoning_content, content_after) = match &msg.reasoning_content {
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
        let reasoning_content = reasoning_content.trim().to_string();
        let content = content_after.trim().to_string();

        let idx = msg_idx as i32;
        buf.special(self.im_start, idx);

        let emit_thinking = (msg_idx as i32) > last_query_index
            || (preserve_thinking && !reasoning_content.is_empty());

        if emit_thinking {
            buf.text("assistant\n", idx)?;
            buf.special(self.think, idx);
            let mut s = String::with_capacity(reasoning_content.len() + 2);
            s.push('\n');
            s.push_str(&reasoning_content);
            s.push('\n');
            buf.text(&s, idx)?;
            buf.special(self.think_end, idx);
            let mut tail = String::with_capacity(content.len() + 2);
            tail.push_str("\n\n");
            tail.push_str(&content);
            buf.text(&tail, idx)?;
        } else {
            let mut s = String::with_capacity(content.len() + 10);
            s.push_str("assistant\n");
            s.push_str(&content);
            buf.text(&s, idx)?;
        }

        for (tc_idx, tc) in msg.tool_calls.iter().enumerate() {
            let name = tc.function.name.as_str();
            // Separator before this tool call
            if tc_idx == 0 {
                if !content.is_empty() {
                    buf.text("\n\n", idx)?;
                }
            } else {
                buf.text("\n", idx)?;
            }

            buf.special(self.tool_call, idx);
            let mut payload = String::with_capacity(name.len() + 32);
            payload.push_str("\n<function=");
            payload.push_str(name);
            payload.push_str(">\n");
            buf.text(&payload, idx)?;

            // Arguments — accept JSON string (decode first) or object
            let args_value = match &tc.function.arguments {
                ToolArguments::Object(v) => v.clone(),
                ToolArguments::Raw(s) => {
                    serde_json::from_str(s).unwrap_or(serde_json::Value::Object(Default::default()))
                }
            };
            if let Some(obj) = args_value.as_object() {
                for (arg_name, arg_value) in obj {
                    let value_str = Self::render_arg_value(arg_value, self.args_as_json);
                    let mut param = String::with_capacity(arg_name.len() + value_str.len() + 24);
                    param.push_str("<parameter=");
                    param.push_str(arg_name);
                    param.push_str(">\n");
                    param.push_str(&value_str);
                    param.push_str("\n</parameter>\n");
                    buf.text(&param, idx)?;
                }
            }

            buf.text("</function>\n", idx)?;
            buf.special(self.tool_call_end, idx);
        }

        buf.special(self.im_end, idx);
        buf.text("\n", idx)?;
        Ok(())
    }

    fn emit_generation_prompt(&self, buf: &mut RenderBuf<'_>) -> Result<(), RenderError> {
        buf.scaffold_special(self.im_start);
        buf.scaffold_text("assistant\n")?;
        if self.enable_thinking {
            buf.scaffold_special(self.think);
            buf.scaffold_text("\n")?;
        } else {
            buf.scaffold_special(self.think);
            buf.scaffold_text("\n\n")?;
            buf.scaffold_special(self.think_end);
            buf.scaffold_text("\n\n")?;
        }
        Ok(())
    }

    fn estimate_capacity(messages: &[Message], tools: Option<&[ToolSpec]>) -> usize {
        let base = messages.len().max(1) * 256;
        let tools_bonus = tools.map(|t| 256 * t.len().max(1) + 512).unwrap_or(0);
        base + tools_bonus
    }
}

impl Renderer for Qwen35Renderer {
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

        let first_is_system = messages[0].role == "system";

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

        let last_qi = Self::last_query_index(messages);

        for (i, msg) in messages.iter().enumerate() {
            let content = Self::render_content_text(&msg.content);
            let content = content.trim();
            match msg.role.as_str() {
                "system" => {
                    if i != 0 {
                        return Err(RenderError::Invalid(
                            "system message must be at the beginning".into(),
                        ));
                    }
                    // Already handled above
                }
                "user" => self.emit_user(&mut buf, content, i as i32)?,
                "assistant" => {
                    let preserve_thinking = should_preserve_past_thinking(
                        messages,
                        i,
                        self.preserve_all_thinking,
                        self.preserve_thinking_between_tool_calls,
                    );
                    self.emit_assistant(&mut buf, msg, i, last_qi, preserve_thinking)?;
                }
                "tool" => self.emit_tool(&mut buf, messages, i, content)?,
                _ => {
                    return Err(RenderError::Invalid(format!(
                        "unexpected message role: {}",
                        msg.role
                    )));
                }
            }
        }

        if add_generation_prompt {
            self.emit_generation_prompt(&mut buf)?;
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

        let mut buf = RenderBuf::new(&self.tokenizer, Self::estimate_capacity(new_messages, None));
        // Trailing newline that the prior render emitted but vLLM stopped on
        buf.scaffold_text("\n")?;

        for (i, msg) in new_messages.iter().enumerate() {
            let content = Self::render_content_text(&msg.content);
            let content = content.trim();
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
                    buf.text("\n", idx)?;
                }
                "tool" => self.emit_tool(&mut buf, new_messages, i, content)?,
                _ => return Ok(None),
            }
        }

        self.emit_generation_prompt(&mut buf)?;

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
// Qwen3.5-VL emits the canonical Qwen-style placeholder block per image:
//     <|vision_start|> + num_tokens × <|image_pad|> + <|vision_end|>
//
// where `num_tokens` is the pre-computed expansion the caller obtained
// from the HF processor (image_grid_thw.prod() / merge_size²). The
// renderer never touches pixel data; `MediaItem::hf_payload` rides
// through as opaque JSON into `MultiModalData::mm_items`.

impl Qwen35Renderer {
    /// Walk the user-message content parts in order, interleaving
    /// placeholder spans where Image / Video parts appear. Mirrors the
    /// HF chat template's behaviour: text and images appear in the
    /// same order the caller listed them in `Content::Parts`.
    ///
    /// `media` items are consumed positionally — the N-th media item
    /// for this message matches the N-th Image/Video part in the
    /// content. Mismatched counts return an `Invalid` error.
    fn emit_user_with_media(
        &self,
        buf: &mut RenderBuf<'_>,
        msg: &Message,
        msg_idx: usize,
        media: &MediaBundle,
        mm: &mut MultiModalData,
    ) -> Result<(), RenderError> {
        let idx = msg_idx as i32;
        buf.special(self.im_start, idx);
        buf.text("user\n", idx)?;

        // Gather this message's media items in render order.
        let mut media_iter = media
            .items
            .iter()
            .filter_map(|(m, item)| (*m == msg_idx).then_some(item));

        match &msg.content {
            crate::types::Content::Null => {
                for item in media_iter.by_ref() {
                    self.emit_media_item(buf, idx, item, mm)?;
                }
            }
            crate::types::Content::Text(s) => {
                // Plain-text user message with attached media: emit
                // images first (canonical Qwen-VL shape:
                // <|vision_start|>...<|vision_end|>{text}), then text.
                for item in media_iter.by_ref() {
                    self.emit_media_item(buf, idx, item, mm)?;
                }
                if !s.is_empty() {
                    buf.text(s.trim(), idx)?;
                }
            }
            crate::types::Content::Parts(parts) => {
                use crate::types::ContentPart;
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            if !text.is_empty() {
                                buf.text(text, idx)?;
                            }
                        }
                        ContentPart::Thinking { .. } => {
                            // Thinking parts shouldn't appear in user
                            // content — silently skip to match the
                            // Python implementation's behaviour.
                        }
                        ContentPart::Image(_) | ContentPart::Video(_) => {
                            let item = media_iter.next().ok_or_else(|| {
                                RenderError::Invalid(format!(
                                    "message {msg_idx} content lists more media parts than the MediaBundle provides"
                                ))
                            })?;
                            self.emit_media_item(buf, idx, item, mm)?;
                        }
                    }
                }
            }
        }

        // Reject extra media items in the bundle that didn't get used —
        // catches off-by-one errors in caller's bundle construction.
        if media_iter.next().is_some() {
            return Err(RenderError::Invalid(format!(
                "MediaBundle has more items for message {msg_idx} than the content's media parts"
            )));
        }

        buf.special(self.im_end, idx);
        buf.text("\n", idx)?;
        Ok(())
    }

    fn emit_media_item(
        &self,
        buf: &mut RenderBuf<'_>,
        idx: i32,
        item: &MediaItem,
        mm: &mut MultiModalData,
    ) -> Result<(), RenderError> {
        let pad = match item.modality {
            Modality::Image => self.image_pad,
            Modality::Video => self.video_pad,
        }
        .ok_or_else(|| {
            RenderError::MissingSpecialToken(match item.modality {
                Modality::Image => "<|image_pad|>".into(),
                Modality::Video => "<|video_pad|>".into(),
            })
        })?;
        let vs = self
            .vision_start
            .ok_or_else(|| RenderError::MissingSpecialToken("<|vision_start|>".into()))?;
        let ve = self
            .vision_end
            .ok_or_else(|| RenderError::MissingSpecialToken("<|vision_end|>".into()))?;

        buf.special(vs, idx);
        let offset = buf.len();
        for _ in 0..item.num_tokens {
            buf.special(pad, idx);
        }
        buf.special(ve, idx);

        // Update MultiModalData. Key by modality string ("image" /
        // "video") so the inference engine glue can route per-key.
        let key = item.modality.as_str().to_string();
        mm.mm_hashes
            .entry(key.clone())
            .or_default()
            .push(item.hash.clone());
        mm.mm_placeholders
            .entry(key.clone())
            .or_default()
            .push(PlaceholderRange {
                offset,
                length: item.num_tokens,
            });
        mm.mm_items
            .entry(key)
            .or_default()
            .push(item.hf_payload.clone());
        Ok(())
    }
}

impl MultimodalRenderer for Qwen35Renderer {
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
        // Fast path: no media → defer to the text-only render.
        if media.is_empty() {
            return self.render(messages, tools, add_generation_prompt);
        }
        if messages.is_empty() {
            return Err(RenderError::EmptyMessages);
        }
        let mut buf = RenderBuf::new(&self.tokenizer, Self::estimate_capacity(messages, tools));
        let first_is_system = messages[0].role == "system";

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

        let last_qi = Self::last_query_index(messages);
        let mut mm = MultiModalData::default();

        for (i, msg) in messages.iter().enumerate() {
            let content = Self::render_content_text(&msg.content);
            let content = content.trim();
            match msg.role.as_str() {
                "system" => {
                    if i != 0 {
                        return Err(RenderError::Invalid(
                            "system message must be at the beginning".into(),
                        ));
                    }
                }
                "user" => {
                    // If this message has attached media OR the caller
                    // provided structured content with image/video parts,
                    // walk the parts inline so order matches the caller's
                    // list. Pure text paths bypass the heavier walk.
                    let has_media = media.items.iter().any(|(idx, _)| *idx == i);
                    let has_structured = matches!(msg.content, crate::types::Content::Parts(_));
                    if has_media || has_structured {
                        self.emit_user_with_media(&mut buf, msg, i, media, &mut mm)?;
                    } else {
                        self.emit_user(&mut buf, content, i as i32)?;
                    }
                }
                "assistant" => {
                    let preserve_thinking = should_preserve_past_thinking(
                        messages,
                        i,
                        self.preserve_all_thinking,
                        self.preserve_thinking_between_tool_calls,
                    );
                    self.emit_assistant(&mut buf, msg, i, last_qi, preserve_thinking)?;
                }
                "tool" => self.emit_tool(&mut buf, messages, i, content)?,
                _ => {
                    return Err(RenderError::Invalid(format!(
                        "unexpected message role: {}",
                        msg.role
                    )));
                }
            }
        }

        if add_generation_prompt {
            self.emit_generation_prompt(&mut buf)?;
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
        // Phase 5a scope: bridge ignores media on the new-turn side
        // (the prior turn's mm_data is carried forward by the caller's
        // glue layer, not by this function). When new_media is
        // non-empty, fall back to a full re-render — bridging
        // image-bearing turns through a verbatim prefix is fragile
        // because placeholder offsets shift if the prior turn was
        // truncated mid-image. Phase 5b can revisit.
        if !new_media.is_empty() {
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
