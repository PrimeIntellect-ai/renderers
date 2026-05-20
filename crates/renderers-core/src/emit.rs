//! Render-buffer helpers used by every hand-coded family.
//!
//! The pattern is the same everywhere: pre-allocated `Vec<u32>` for tokens
//! and `Vec<i32>` for per-token message attribution, with three primitives
//! to fill them. Centralising the primitives lets each family stay focused
//! on its own template logic without re-deriving the bookkeeping.

use crate::tokenizer::Tokenizer;
use crate::types::{RenderError, RenderedTokens, SCAFFOLD_IDX};

/// Mutable render-time buffer paired with a tokenizer reference.
///
/// Holds both the token stream and the parallel `message_indices` array.
/// All emits are O(1) amortised against the pre-allocated capacity.
pub struct RenderBuf<'tok> {
    tokens: Vec<u32>,
    indices: Vec<i32>,
    tokenizer: &'tok Tokenizer,
    /// Scratch `Vec` reused across `encode` calls so each text segment
    /// doesn't allocate. The tokenizer's `encode` API returns its own
    /// `Encoding`, so the saving is at the buffer-extension layer, not
    /// at encode itself.
    scratch_offsets: Vec<usize>,
}

impl<'tok> std::fmt::Debug for RenderBuf<'tok> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderBuf")
            .field("tokens_len", &self.tokens.len())
            .field("indices_len", &self.indices.len())
            .finish()
    }
}

impl<'tok> RenderBuf<'tok> {
    pub fn new(tokenizer: &'tok Tokenizer, hint: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(hint),
            indices: Vec::with_capacity(hint),
            tokenizer,
            scratch_offsets: Vec::new(),
        }
    }

    #[inline]
    pub fn tokenizer(&self) -> &Tokenizer {
        self.tokenizer
    }

    /// Append a single special token id to the buffer.
    #[inline]
    pub fn special(&mut self, token_id: u32, msg_idx: i32) {
        self.tokens.push(token_id);
        self.indices.push(msg_idx);
    }

    /// Append a span of token ids to the buffer, all attributed to the
    /// same message index.
    #[inline]
    pub fn ids(&mut self, token_ids: &[u32], msg_idx: i32) {
        self.tokens.extend_from_slice(token_ids);
        // `resize` with a Copy fill is the cheapest way to extend the
        // indices vector by N elements of the same value.
        let new_len = self.indices.len() + token_ids.len();
        self.indices.resize(new_len, msg_idx);
    }

    /// Encode `text` and append the resulting tokens, attributing all of
    /// them to `msg_idx`. Empty strings are a no-op (saves a tokenizer
    /// call on the common "no content here" path).
    #[inline]
    pub fn text(&mut self, text: &str, msg_idx: i32) -> Result<(), RenderError> {
        if text.is_empty() {
            return Ok(());
        }
        let encoded = self.tokenizer.encode_no_special(text)?;
        self.ids(encoded.as_slice(), msg_idx);
        Ok(())
    }

    /// Append a scaffold token (one whose attribution is "structural,
    /// not from any message" — uses [`SCAFFOLD_IDX`]).
    #[inline]
    pub fn scaffold_special(&mut self, token_id: u32) {
        self.special(token_id, SCAFFOLD_IDX);
    }

    /// Encode `text` and append as scaffolding (attribution [`SCAFFOLD_IDX`]).
    #[inline]
    pub fn scaffold_text(&mut self, text: &str) -> Result<(), RenderError> {
        self.text(text, SCAFFOLD_IDX)
    }

    /// Consume the buffer and return a [`RenderedTokens`].
    pub fn into_rendered(self) -> RenderedTokens {
        debug_assert_eq!(self.tokens.len(), self.indices.len());
        let _ = self.scratch_offsets; // keep the field but ignore
        RenderedTokens {
            token_ids: self.tokens,
            message_indices: self.indices,
            multi_modal_data: None,
        }
    }

    /// Take the token ids only, dropping per-token attribution. Used by
    /// `render_ids` callers that don't need the indices array.
    pub fn into_token_ids(self) -> Vec<u32> {
        self.tokens
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
