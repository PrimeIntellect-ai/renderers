//! DefaultRenderer — Jinja-template fallback for models without a
//! hand-coded family.
//!
//! Port of `renderers/default.py`. Two key differences from the Python
//! implementation:
//!
//! - Renders the template with [`minijinja`] (vs HF's Python Jinja). The
//!   `chat_template` string is loaded from the model's
//!   `tokenizer_config.json` and rendered against a context built from
//!   the messages + tools. minijinja covers the Jinja2 subset HF
//!   templates actually use (`for`, `if`, `set`, filters like `tojson`,
//!   `length`, `trim`); anything more exotic will return a render error
//!   instead of silently miscompiling.
//! - Per-token attribution is incremental: render the conversation
//!   prefix-by-prefix and attribute the delta to each message index.
//!   Same algorithm as the Python class, but driven by minijinja
//!   instead of HF's `apply_chat_template`.
//!
//! `parse_response` is intentionally basic: strip stop tokens, decode,
//! split on `</think>` if present. Models with structured tool calls
//! need a hand-coded family — DefaultRenderer doesn't try to guess.
//!
//! `bridge_to_next_turn` returns `None` unconditionally: without
//! template-specific knowledge of the turn-close token, the bridge
//! contract can't be proven, so the caller falls back to a full
//! re-render.

use std::sync::Arc;

use minijinja::Environment;
use minijinja::value::Value as MjValue;
use serde_json::Value as JsonValue;

use crate::tokenizer::Tokenizer;
use crate::traits::Renderer;
use crate::types::{
    Message, ParsedResponse, RenderError, RenderedTokens, SCAFFOLD_IDX, ToolArguments, ToolSpec,
};

/// Builder for [`DefaultRenderer`].
pub struct DefaultRendererBuilder {
    chat_template: String,
    stop_token_ids: Vec<u32>,
    extra_context: Vec<(String, JsonValue)>,
}

impl DefaultRendererBuilder {
    pub fn new(chat_template: impl Into<String>) -> Self {
        Self {
            chat_template: chat_template.into(),
            stop_token_ids: Vec::new(),
            extra_context: Vec::new(),
        }
    }
    /// Stop tokens — typically `[eos_token_id]`. The caller decides; the
    /// renderer doesn't probe the tokenizer for `eos_token` since the
    /// canonical id varies per model.
    pub fn stop_token_ids(mut self, ids: Vec<u32>) -> Self {
        self.stop_token_ids = ids;
        self
    }
    /// Add a `key=value` context variable for the Jinja template.
    /// Common entries: `bos_token`, `eos_token`, `add_generation_prompt`.
    pub fn add_context(mut self, key: impl Into<String>, value: JsonValue) -> Self {
        self.extra_context.push((key.into(), value));
        self
    }
    pub fn build(self, tokenizer: Tokenizer) -> Result<DefaultRenderer, RenderError> {
        DefaultRenderer::new_with(tokenizer, self)
    }
}

impl std::fmt::Debug for DefaultRendererBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultRendererBuilder")
            .field("chat_template_len", &self.chat_template.len())
            .field("stop_token_ids", &self.stop_token_ids)
            .field("extra_context_keys", &self.extra_context.len())
            .finish()
    }
}

pub struct DefaultRenderer {
    tokenizer: Tokenizer,
    env: Arc<Environment<'static>>,
    extra_context: Vec<(String, JsonValue)>,
    stop_token_ids: Vec<u32>,
}

impl std::fmt::Debug for DefaultRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultRenderer")
            .field("stop_token_ids", &self.stop_token_ids)
            .field("extra_context_keys", &self.extra_context.len())
            .finish_non_exhaustive()
    }
}

impl Clone for DefaultRenderer {
    fn clone(&self) -> Self {
        Self {
            tokenizer: self.tokenizer.clone(),
            env: self.env.clone(),
            extra_context: self.extra_context.clone(),
            stop_token_ids: self.stop_token_ids.clone(),
        }
    }
}

impl DefaultRenderer {
    fn new_with(tokenizer: Tokenizer, cfg: DefaultRendererBuilder) -> Result<Self, RenderError> {
        let mut env = Environment::new();
        // HF chat templates use whitespace-stripped markers freely
        // (e.g. `{%- if foo -%}`); minijinja respects that via the
        // `lstrip_blocks` / `trim_blocks` knobs below.
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        env.add_template_owned("chat", cfg.chat_template)
            .map_err(|e| RenderError::Invalid(format!("chat_template parse: {e}")))?;
        Ok(Self {
            tokenizer,
            env: Arc::new(env),
            extra_context: cfg.extra_context,
            stop_token_ids: cfg.stop_token_ids,
        })
    }

    pub fn builder(chat_template: impl Into<String>) -> DefaultRendererBuilder {
        DefaultRendererBuilder::new(chat_template)
    }

    /// Render the template up to `messages[..end]` (exclusive). When
    /// `add_generation_prompt` is true the template's gen-prompt branch
    /// fires.
    fn render_jinja(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<String, RenderError> {
        // Build a single flat context map up front. minijinja's
        // `context!` macro and `Value::from_object` produce equivalent
        // results, but a single dict keeps the per-render allocation
        // count constant regardless of how many extra context keys the
        // caller passes (vs the wrapped-Object chain previously used).
        let mut ctx_map = serde_json::Map::new();
        ctx_map.insert(
            "messages".into(),
            serde_json::to_value(messages_to_value(messages)?).unwrap_or(JsonValue::Null),
        );
        let tools_value: MjValue = match tools {
            Some(t) => tools_to_value(t),
            None => MjValue::from(Vec::<MjValue>::new()),
        };
        ctx_map.insert(
            "tools".into(),
            serde_json::to_value(tools_value).unwrap_or(JsonValue::Null),
        );
        ctx_map.insert(
            "add_generation_prompt".into(),
            JsonValue::Bool(add_generation_prompt),
        );
        for (k, v) in &self.extra_context {
            ctx_map.insert(k.clone(), v.clone());
        }
        let ctx = MjValue::from_serialize(JsonValue::Object(ctx_map));

        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| RenderError::Invalid(format!("chat_template lookup: {e}")))?;
        tmpl.render(ctx)
            .map_err(|e| RenderError::Invalid(format!("chat_template render: {e}")))
    }

    fn encode_full(&self, text: &str) -> Result<Vec<u32>, RenderError> {
        Ok(self.tokenizer.encode_no_special(text)?.as_slice().to_vec())
    }
}

impl Renderer for DefaultRenderer {
    fn render(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError> {
        if messages.is_empty() {
            return Err(RenderError::EmptyMessages);
        }
        // Incremental render: tokenise prefix-by-prefix, attribute the
        // delta to each message index. Same approach as the Python class.
        let mut token_ids: Vec<u32> = Vec::new();
        let mut message_indices: Vec<i32> = Vec::new();
        let mut prev_len = 0usize;

        for (i, _) in messages.iter().enumerate() {
            let text = self.render_jinja(&messages[..=i], tools, false)?;
            let ids = self.encode_full(&text)?;
            if ids.len() < prev_len {
                // Template didn't extend prefix-monotonically — fall back to
                // a single full render attributed entirely to scaffolding.
                let all = self.encode_full(&self.render_jinja(
                    messages,
                    tools,
                    add_generation_prompt,
                )?)?;
                return Ok(RenderedTokens {
                    token_ids: all.clone(),
                    message_indices: vec![SCAFFOLD_IDX; all.len()],
                    multi_modal_data: None,
                });
            }
            let new_count = ids.len() - prev_len;
            message_indices.extend(std::iter::repeat_n(i as i32, new_count));
            token_ids = ids;
            prev_len = token_ids.len();
        }

        if add_generation_prompt {
            let full = self.render_jinja(messages, tools, true)?;
            let full_ids = self.encode_full(&full)?;
            if full_ids.len() >= prev_len {
                let gen_count = full_ids.len() - prev_len;
                message_indices.extend(std::iter::repeat_n(SCAFFOLD_IDX, gen_count));
                token_ids = full_ids;
            } else {
                token_ids = full_ids;
                message_indices.truncate(token_ids.len());
            }
        }

        Ok(RenderedTokens {
            token_ids,
            message_indices,
            multi_modal_data: None,
        })
    }

    fn render_ids(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>, RenderError> {
        // Fast path: one full render instead of N prefix renders. Used by
        // callers that don't need per-token attribution.
        let text = self.render_jinja(messages, tools, add_generation_prompt)?;
        self.encode_full(&text)
    }

    fn parse_response(&self, token_ids: &[u32]) -> ParsedResponse {
        // Truncate at the first stop token.
        let end = token_ids
            .iter()
            .position(|t| self.stop_token_ids.contains(t))
            .unwrap_or(token_ids.len());
        let text = self.tokenizer.decode(&token_ids[..end]).unwrap_or_default();

        // Split out a `<think>...</think>` block if present. Same logic
        // as the Python fallback.
        let (reasoning_content, content) = match text.split_once("</think>") {
            Some((before, after)) => {
                let r = if let Some((_, inner)) = before.rsplit_once("<think>") {
                    inner.to_string()
                } else {
                    before.to_string()
                };
                (Some(r).filter(|s| !s.is_empty()), after.to_string())
            }
            None => (None, text.clone()),
        };

        ParsedResponse {
            content,
            reasoning_content,
            tool_calls: Vec::new(),
        }
    }

    fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }

    fn bridge_to_next_turn(
        &self,
        _previous_prompt_ids: &[u32],
        _previous_completion_ids: &[u32],
        _new_messages: &[Message],
        _tools: Option<&[ToolSpec]>,
    ) -> Result<Option<RenderedTokens>, RenderError> {
        // Same contract as the Python DefaultRenderer: without family
        // knowledge of the turn-close token, the bridge can't be proven.
        Ok(None)
    }
}

// ── Jinja context conversion ──────────────────────────────────────────

fn messages_to_value(messages: &[Message]) -> Result<MjValue, RenderError> {
    let mut out: Vec<MjValue> = Vec::with_capacity(messages.len());
    for m in messages {
        let mut map = serde_json::Map::new();
        map.insert("role".into(), JsonValue::String(m.role.clone()));
        // Content: string fast-path, structured parts pass through as JSON
        let content_value = match &m.content {
            crate::types::Content::Null => JsonValue::Null,
            crate::types::Content::Text(s) => JsonValue::String(s.clone()),
            crate::types::Content::Parts(parts) => serde_json::to_value(parts)
                .map_err(|e| RenderError::Invalid(format!("content serialisation: {e}")))?,
        };
        map.insert("content".into(), content_value);
        if let Some(name) = &m.name {
            map.insert("name".into(), JsonValue::String(name.clone()));
        }
        if let Some(tcid) = &m.tool_call_id {
            map.insert("tool_call_id".into(), JsonValue::String(tcid.clone()));
        }
        if let Some(r) = &m.reasoning_content {
            map.insert("reasoning_content".into(), JsonValue::String(r.clone()));
        }
        if !m.tool_calls.is_empty() {
            let tcs: Vec<JsonValue> = m
                .tool_calls
                .iter()
                .map(|tc| {
                    let args = match &tc.function.arguments {
                        ToolArguments::Object(v) => v.clone(),
                        ToolArguments::Raw(s) => {
                            serde_json::from_str(s).unwrap_or(JsonValue::String(s.clone()))
                        }
                    };
                    serde_json::json!({
                        "type": tc.kind,
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": args,
                        },
                    })
                })
                .collect();
            map.insert("tool_calls".into(), JsonValue::Array(tcs));
        }
        out.push(MjValue::from_serialize(JsonValue::Object(map)));
    }
    Ok(MjValue::from(out))
}

fn tools_to_value(tools: &[ToolSpec]) -> MjValue {
    let mut out: Vec<MjValue> = Vec::with_capacity(tools.len());
    for t in tools {
        let v = serde_json::json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        });
        out.push(MjValue::from_serialize(v));
    }
    MjValue::from(out)
}
