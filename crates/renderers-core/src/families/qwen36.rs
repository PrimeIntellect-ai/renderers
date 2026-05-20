//! Qwen3.6 renderer. Delta vs Qwen3.5: tool-call arguments serialise
//! through JSON (bools → `true`/`false`, None → `null`, etc.) instead of
//! Python `str()`. Everything else — template structure, parser,
//! tool-call XML, thinking markers, bridge logic — is identical to
//! Qwen3.5, so this is a one-line config delta on the Qwen3.5 builder.
//!
//! Mirrors `renderers/qwen36.py`.

use crate::families::Qwen35RendererBuilder;

/// Build a Qwen3.6 renderer.
///
/// Type alias preserved as a re-export of [`Qwen35Renderer`](crate::families::Qwen35Renderer)
/// — the type system doesn't distinguish them at runtime; they differ
/// only in the `args_as_json` flag. The builder below is the right
/// public surface.
pub use crate::families::Qwen35Renderer as Qwen36Renderer;

/// Builder for [`Qwen36Renderer`] (a Qwen3.5 with JSON-flavoured tool
/// arguments).
#[derive(Debug, Clone, Default)]
pub struct Qwen36RendererBuilder {
    inner: Qwen35RendererBuilder,
}

impl Qwen36RendererBuilder {
    pub fn enable_thinking(mut self, on: bool) -> Self {
        self.inner = self.inner.enable_thinking(on);
        self
    }
    pub fn preserve_all_thinking(mut self, on: bool) -> Self {
        self.inner = self.inner.preserve_all_thinking(on);
        self
    }
    pub fn preserve_thinking_between_tool_calls(mut self, on: bool) -> Self {
        self.inner = self.inner.preserve_thinking_between_tool_calls(on);
        self
    }
    pub fn build(
        self,
        tokenizer: crate::tokenizer::Tokenizer,
    ) -> Result<Qwen36Renderer, crate::types::RenderError> {
        self.inner.args_as_json(true).build(tokenizer)
    }
}
