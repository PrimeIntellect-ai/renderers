//! Core data types for renderers.
//!
//! The shapes mirror the Python `renderers.base` types so JSON round-trips
//! and PyO3 wrapping stay mechanical. Strings are owned (`String`) — PyO3
//! always materialises strings on entry, so `Cow<'a, str>` would only
//! propagate lifetimes for no win. The few `&str` borrows that pay off are
//! taken locally inside renderer implementations from `&[Message]` slices.

use std::ops::Range;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Sentinel value for `message_indices` entries that come from structural
/// scaffolding rather than a specific message (e.g. the generation prompt).
///
/// Kept as a named constant so the `-1` in code is searchable and easy to
/// audit. Matches the Python contract at `renderers/base.py:160`.
pub const SCAFFOLD_IDX: i32 = -1;

/// A single content part inside a multi-part message body.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text.
    Text { text: String },
    /// Model chain-of-thought as a content part.
    Thinking { thinking: String },
    /// Image reference. Resolution to bytes / processor output happens
    /// in the multimodal renderer.
    Image(ImageRef),
    /// Video reference; mirrors [`ImageRef`].
    Video(VideoRef),
}

/// Image source variants accepted in [`ContentPart::Image`]. Phase 1
/// covers text-only families, so only the URL/path discriminators carry
/// data — inline bytes are routed through `serde_json::Value` payload
/// for now and resolved by the (Phase 5) multimodal port.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ImageRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// Video source variants accepted in [`ContentPart::Video`].
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct VideoRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// Message body. Either a plain string or a list of structured parts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Content {
    Null,
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Default for Content {
    fn default() -> Self {
        Content::Text(String::new())
    }
}

impl Content {
    /// Borrow the body as a `&str` if it is a plain string; returns
    /// `""` for `Parts` variants (Qwen3 ignores list content entirely).
    pub fn as_text(&self) -> &str {
        match self {
            Content::Null => "",
            Content::Text(s) => s.as_str(),
            Content::Parts(_) => "",
        }
    }

    pub fn as_text_or_none_literal(&self) -> &str {
        match self {
            Content::Null => "None",
            Content::Text(s) => s.as_str(),
            Content::Parts(_) => "",
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Content::Null => true,
            Content::Text(s) => s.is_empty(),
            Content::Parts(p) => p.is_empty(),
        }
    }
}

/// Function body inside a [`ToolCall`].
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolCallFunction {
    #[serde(default)]
    pub name: String,
    /// Arguments may arrive as a JSON object or as a pre-serialised JSON
    /// string (some OpenAI-format clients do this); preserve the
    /// distinction.
    #[serde(default)]
    pub arguments: ToolArguments,
}

/// Structured tool invocation in OpenAI function-calling format.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    #[serde(default = "default_tool_type", rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: ToolCallFunction,
}

fn default_tool_type() -> String {
    "function".to_string()
}

/// Tool specification (OpenAI function-calling format) passed to
/// [`Renderer::render`](crate::Renderer::render).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
    #[serde(default, skip)]
    pub openai_envelope: bool,
}

/// A single turn in a multi-turn conversation.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Content,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    /// Borrow `content` as a `&str` only when it is a plain string. Many
    /// hand-coded renderers (Qwen3, GLM5, ...) drop list-content entirely
    /// for non-multimodal text paths; this helper makes that explicit.
    #[inline]
    pub fn text_content(&self) -> &str {
        self.content.as_text()
    }

    #[inline]
    pub fn visible_text_content(&self) -> &str {
        self.content.as_text_or_none_literal()
    }
}

/// Tool-call argument payload. The JSON-object case is the common path;
/// the raw-string case preserves the OpenAI quirk where some clients
/// pre-serialise arguments to a string.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolArguments {
    Raw(String),
    Object(serde_json::Value),
}

impl Default for ToolArguments {
    fn default() -> Self {
        ToolArguments::Object(serde_json::Value::Object(Default::default()))
    }
}

impl ToolArguments {
    /// Render arguments as a JSON string suitable for inserting verbatim
    /// into a tool-call payload (matches Python's
    /// `json.dumps(arguments, ensure_ascii=False)`).
    pub fn to_json_string(&self) -> String {
        match self {
            ToolArguments::Raw(s) => s.clone(),
            ToolArguments::Object(v) => {
                serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string())
            }
        }
    }
}

/// Where a single multimodal item's placeholder tokens sit in the stream.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlaceholderRange {
    pub offset: usize,
    pub length: usize,
}

/// Multimodal sidecar emitted alongside the token stream. The shape
/// mirrors vLLM's `mm_*` payload without depending on vLLM types.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct MultiModalData {
    #[serde(default)]
    pub mm_hashes: std::collections::BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub mm_placeholders: std::collections::BTreeMap<String, Vec<PlaceholderRange>>,
    /// Per-item processor outputs. The values are passed through as opaque
    /// JSON to keep this crate framework-agnostic; vision processors live
    /// behind the PyO3 boundary in the current Phase 1 design.
    #[serde(default)]
    pub mm_items: std::collections::BTreeMap<String, Vec<serde_json::Value>>,
}

impl MultiModalData {
    pub fn is_empty(&self) -> bool {
        self.mm_hashes.is_empty() && self.mm_placeholders.is_empty() && self.mm_items.is_empty()
    }
}

/// Result of rendering messages to tokens.
///
/// `token_ids` and `message_indices` are parallel: `message_indices[i]` is
/// the index into the input `messages` slice of the message that produced
/// `token_ids[i]`, or [`SCAFFOLD_IDX`] for structural scaffolding tokens.
///
/// Both vectors are sized once during render — see
/// [`RenderedTokens::with_capacity`].
#[derive(Clone, Debug, Default)]
pub struct RenderedTokens {
    pub token_ids: Vec<u32>,
    pub message_indices: Vec<i32>,
    pub multi_modal_data: Option<MultiModalData>,
}

impl RenderedTokens {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocate both buffers to the same capacity. Renderers pass an
    /// estimate based on `messages.len() * 256` to keep the hot path
    /// realloc-free for typical conversations.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            token_ids: Vec::with_capacity(cap),
            message_indices: Vec::with_capacity(cap),
            multi_modal_data: None,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

/// Per-attempt outcome of parsing a single tool-call block. Matches the
/// Python `ToolCallParseStatus` semantics — every parse attempt surfaces
/// (success and malformed alike), distinguished by this status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallParseStatus {
    Ok,
    InvalidJson,
    UnclosedBlock,
    MissingName,
    MalformedStructure,
}

impl ToolCallParseStatus {
    /// Wire string matching the Python enum values
    /// (`"ok" | "invalid_json" | ...`) so PyO3 can round-trip them.
    pub fn as_wire(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::InvalidJson => "invalid_json",
            Self::UnclosedBlock => "unclosed_block",
            Self::MissingName => "missing_name",
            Self::MalformedStructure => "malformed_structure",
        }
    }
}

/// A single tool-call block as the parser saw it.
///
/// `arguments` carries `None` only when the block was so malformed that
/// nothing could be recovered; successful parses produce
/// [`ToolArguments::Object`], pre-serialised string arguments produce
/// [`ToolArguments::Raw`].
#[derive(Clone, Debug)]
pub struct ParsedToolCall {
    pub raw: String,
    pub name: Option<String>,
    pub arguments: Option<ToolArguments>,
    /// Half-open `[start, end)` slice into the stop-stripped completion
    /// token stream. `None` for text-based parsers that can't cheaply
    /// recover offsets.
    pub token_span: Option<Range<usize>>,
    pub status: ToolCallParseStatus,
    /// Native id when the format carries one (Kimi K2).
    pub id: Option<String>,
}

impl Default for ParsedToolCall {
    fn default() -> Self {
        Self {
            raw: String::new(),
            name: None,
            arguments: None,
            token_span: None,
            status: ToolCallParseStatus::Ok,
            id: None,
        }
    }
}

/// Result of parsing completion tokens back into a structured message.
#[derive(Clone, Debug, Default)]
pub struct ParsedResponse {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ParsedToolCall>,
}

/// Errors surfaced by rendering.
#[derive(Debug, Error)]
pub enum RenderError {
    #[error("no messages provided")]
    EmptyMessages,
    #[error("special token {0:?} not found in tokenizer vocabulary")]
    MissingSpecialToken(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("invalid input: {0}")]
    Invalid(String),
}
