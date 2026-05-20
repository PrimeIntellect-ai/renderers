//! Tokenizer-path → renderer factory registry.
//!
//! Mirrors `renderers/base.py:MODEL_RENDERER_MAP` for the subset of
//! families ported to Rust so far. New families slot in by adding a
//! match arm in [`create_renderer`].

use crate::families::{Qwen35Renderer, Qwen3Renderer};
use crate::tokenizer::Tokenizer;
use crate::traits::Renderer;
use crate::types::RenderError;

/// Renderer family identifier — closed enum used by [`create_renderer`].
/// Adding a family means a new variant here plus a match arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererKind {
    Qwen3,
    Qwen35,
}

impl RendererKind {
    pub fn from_str(name: &str) -> Option<Self> {
        match name {
            "qwen3" | "Qwen3" => Some(Self::Qwen3),
            "qwen35" | "qwen3.5" | "Qwen3.5" => Some(Self::Qwen35),
            _ => None,
        }
    }
}

/// Configuration passed to [`create_renderer`].
#[derive(Clone, Debug, Default)]
pub struct RendererConfig {
    pub preserve_all_thinking: bool,
    pub preserve_thinking_between_tool_calls: bool,
    /// `None` keeps the family default; the Qwen3.5 Python shim probes
    /// the tokenizer's Jinja template to pick the right polarity and
    /// forwards the result here so the Rust side stays template-agnostic.
    pub enable_thinking: Option<bool>,
}

/// Build a renderer of the requested kind backed by `tokenizer`.
pub fn create_renderer(
    kind: RendererKind,
    tokenizer: Tokenizer,
    cfg: RendererConfig,
) -> Result<Box<dyn Renderer>, RenderError> {
    match kind {
        RendererKind::Qwen3 => Ok(Box::new(
            Qwen3Renderer::builder()
                .preserve_all_thinking(cfg.preserve_all_thinking)
                .preserve_thinking_between_tool_calls(cfg.preserve_thinking_between_tool_calls)
                .build(tokenizer)?,
        )),
        RendererKind::Qwen35 => {
            let mut b = Qwen35Renderer::builder()
                .preserve_all_thinking(cfg.preserve_all_thinking)
                .preserve_thinking_between_tool_calls(cfg.preserve_thinking_between_tool_calls);
            if let Some(en) = cfg.enable_thinking {
                b = b.enable_thinking(en);
            }
            Ok(Box::new(b.build(tokenizer)?))
        }
    }
}
