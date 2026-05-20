//! The [`Renderer`] trait and its multimodal extension.
//!
//! Both are object-safe so a `Box<dyn Renderer>` (or `Arc<dyn Renderer>`)
//! at the public boundary works without extra ceremony. Family-specific
//! configuration lives on the concrete struct that impls these traits.

use crate::types::{MediaBundle, MultiModalData, ParsedResponse, RenderError, RenderedTokens};
use crate::types::{Message, ToolSpec};

/// Deterministic message → token renderer for a specific model family.
///
/// Implementors must:
///
/// - Be `Send + Sync` so a single instance can be shared via `Arc` across
///   threads (the Python `RendererPool` is obsolete in Rust).
/// - Produce byte-for-byte identical output to the corresponding Python
///   renderer for the same inputs — verified by the `test_render_ids`,
///   `test_bridge`, `test_roundtrip`, and `test_parse_response_robustness`
///   golden suites.
pub trait Renderer: Send + Sync + std::fmt::Debug {
    /// Render `messages` to tokens with per-token message attribution.
    fn render(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError>;

    /// Render `messages` to tokens, dropping per-token attribution. The
    /// default impl delegates to [`Renderer::render`]; family-specific
    /// renderers may override with a slimmer path if it shows up in
    /// profiling (the saving is one `Vec<i32>` allocation).
    fn render_ids(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>, RenderError> {
        Ok(self
            .render(messages, tools, add_generation_prompt)?
            .token_ids)
    }

    /// Parse a completion's token ids back into a structured response.
    fn parse_response(&self, token_ids: &[u32]) -> ParsedResponse;

    /// Stop token ids the sampler should respect.
    fn stop_token_ids(&self) -> &[u32];

    /// Extend the prior turn's tokens verbatim with `new_messages`.
    ///
    /// Contract:
    /// - The returned token stream starts with
    ///   `previous_prompt_ids + previous_completion_ids` (byte-for-byte).
    /// - Returns `None` if `new_messages` contains an assistant turn
    ///   (refuses to retokenize sampled output) or if the prior turn was
    ///   truncated and no canonical close can be synthesised.
    fn bridge_to_next_turn(
        &self,
        previous_prompt_ids: &[u32],
        previous_completion_ids: &[u32],
        new_messages: &[Message],
        tools: Option<&[ToolSpec]>,
    ) -> Result<Option<RenderedTokens>, RenderError>;

    /// Downcast to a multimodal renderer if this implementor supports it.
    /// Default returns `None`; multimodal families override.
    fn as_multimodal(&self) -> Option<&dyn MultimodalRenderer> {
        None
    }
}

/// Extension implemented by multimodal-capable renderers.
///
/// Phase 5 design: the renderer **does not touch raw pixel data**. The
/// caller resolves image/video parts upstream (via the HF processor in
/// the Phase 5a Python shim, or a candle-backed [`MediaResolver`] in
/// Phase 5b) and hands the renderer a [`MediaBundle`] with each item's
/// placeholder count pre-computed.
///
/// Concrete implementors are added in Phase 5a; this trait surface is
/// frozen now so that diff is purely additive on a stable API.
pub trait MultimodalRenderer: Renderer {
    /// Placeholder token id → modality marker (1 = image, 2 = video).
    /// Used by the trainer to build per-token `mm_type_ids` masks.
    fn mm_token_type_id_map(&self) -> &[(u32, u8)];

    /// Render `messages` with pre-resolved `media`.
    ///
    /// The renderer walks `messages` and pulls items from `media` in
    /// order. Each `MediaItem.num_tokens` is the count of placeholder
    /// tokens the renderer must emit between the modality's
    /// start/end special tokens. The item's `hf_payload` rides through
    /// as opaque data on [`RenderedTokens::multi_modal_data`].
    fn render_with_media(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSpec]>,
        media: &MediaBundle,
        add_generation_prompt: bool,
    ) -> Result<RenderedTokens, RenderError>;

    /// Multimodal-aware bridge. Same contract as
    /// [`Renderer::bridge_to_next_turn`] plus `new_media` for the
    /// extension and `previous_multi_modal_data` so prior placeholders
    /// (and their hashes / payloads) survive across turns.
    fn bridge_to_next_turn_with_media(
        &self,
        previous_prompt_ids: &[u32],
        previous_completion_ids: &[u32],
        new_messages: &[Message],
        tools: Option<&[ToolSpec]>,
        new_media: &MediaBundle,
        previous_multi_modal_data: Option<&MultiModalData>,
    ) -> Result<Option<RenderedTokens>, RenderError>;
}

/// Resolves raw image / video sources to processor outputs.
///
/// Phase 5a uses a Python-side implementation that wraps HF's
/// `Qwen3VLImageProcessor` / `KimiVLImageProcessor` and delivers
/// [`MediaItem`]s pre-sized. Phase 5b will add a Rust-native
/// implementation backed by `candle` (or `ort`) so downstream Rust
/// callers can skip the Python boundary entirely.
///
/// The trait is deliberately tiny: a single resolve call per item,
/// caller chooses the modality and source.
pub trait MediaResolver: Send + Sync + std::fmt::Debug {
    /// Resolve a single source (URL / filesystem path / inline bytes)
    /// to a sized [`MediaItem`]. Implementations are free to cache by
    /// hash; the resolver lives for the lifetime of a renderer pool
    /// slot.
    fn resolve_image(
        &self,
        source: &MediaSource<'_>,
    ) -> Result<crate::types::MediaItem, RenderError>;

    /// Resolve a video source — Phase 5b only. The default impl returns
    /// an error so Phase 5a callers don't accidentally pass through.
    fn resolve_video(
        &self,
        _source: &MediaSource<'_>,
    ) -> Result<crate::types::MediaItem, RenderError> {
        Err(RenderError::Invalid(
            "video resolution not implemented in this resolver".into(),
        ))
    }
}

/// A source descriptor for a media item the caller wants resolved.
#[derive(Clone, Debug)]
pub enum MediaSource<'a> {
    Url(&'a str),
    Path(&'a std::path::Path),
    /// Inline image bytes (PNG / JPEG / WebP / etc.). The resolver
    /// detects the format from the bytes themselves.
    Bytes(&'a [u8]),
}
