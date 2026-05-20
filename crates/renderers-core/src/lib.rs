//! `renderers-core` — deterministic message → token rendering for LLM
//! training and inference.
//!
//! This crate is the pure-Rust foundation: data types, the [`Renderer`]
//! trait, parsing primitives, and per-family renderer implementations.
//! The Python wrapper lives in `renderers-py`.
//!
//! # Design at a glance
//!
//! - Messages flow into a [`Renderer`] which emits [`RenderedTokens`]
//!   (token ids plus per-token message attribution).
//! - Completion token ids flow back into [`Renderer::parse_response`],
//!   which returns a [`ParsedResponse`] with content, optional reasoning,
//!   and per-attempt [`ParsedToolCall`] records (success and malformed
//!   both surface, distinguished by [`ToolCallParseStatus`]).
//! - Multi-turn rollouts use [`Renderer::bridge_to_next_turn`] to extend
//!   the prior turn's token stream byte-for-byte, avoiding re-tokenization
//!   drift.
//!
//! The crate is `#![forbid(unsafe_code)]` and aims to keep allocation off
//! the hot path: render buffers grow once, parsing uses a per-call arena,
//! and concrete renderers cache resolved special-token ids at construction.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

pub mod bridge;
pub mod emit;
pub mod families;
pub(crate) mod json;
pub mod parsing;
pub mod processing;
pub mod registry;
pub mod thinking;
pub mod tokenizer;
pub mod traits;
pub mod types;

pub use traits::{MediaResolver, MediaSource, MultimodalRenderer, Renderer};
pub use types::{
    Content, ContentPart, ImageRef, MediaBundle, MediaItem, Message, Modality, MultiModalData,
    ParsedResponse, ParsedToolCall, PlaceholderRange, RenderError, RenderedTokens, SCAFFOLD_IDX,
    ToolArguments, ToolCall, ToolCallFunction, ToolCallParseStatus, ToolSpec, VideoRef,
};
