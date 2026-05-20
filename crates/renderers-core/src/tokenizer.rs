//! Thin wrapper around `tokenizers::Tokenizer`.
//!
//! Provides three things the bare crate doesn't:
//! 1. A cached `unk_token_id` lookup so [`Tokenizer::token_to_id_strict`]
//!    can match Python's "unk-id-is-missing" convention.
//! 2. An `encode_no_special` that returns `Vec<u32>` directly, sized to
//!    the encoding length — saves the caller from juggling the
//!    `tokenizers::Encoding` struct on every hot-path text segment.
//! 3. `Send + Sync` Arc-friendly storage so renderers can share one
//!    instance across threads.

use std::sync::Arc;

use crate::types::RenderError;

/// Owned tokenizer handle. Cloning is cheap (`Arc<Inner>`); the
/// `tokenizers::Tokenizer` itself is held behind the Arc.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    tok: tokenizers::Tokenizer,
    unk_id: Option<u32>,
}

impl Tokenizer {
    /// Load a `tokenizer.json` from disk.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, RenderError> {
        let tok = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| RenderError::Tokenizer(e.to_string()))?;
        Ok(Self::wrap(tok))
    }

    /// Wrap an already-loaded `tokenizers::Tokenizer`.
    pub fn wrap(tok: tokenizers::Tokenizer) -> Self {
        let unk_id = tok.token_to_id("<unk>");
        Self {
            inner: Arc::new(Inner { tok, unk_id }),
        }
    }

    /// Returns the token id for `token`, or `None` if missing /
    /// resolved to `<unk>`. Matches the Python helper at
    /// `renderers/parsers.py:_token_id`.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        let tid = self.inner.tok.token_to_id(token)?;
        if Some(tid) == self.inner.unk_id {
            None
        } else {
            Some(tid)
        }
    }

    /// Strict variant: returns an error if the token is missing.
    pub fn token_to_id_strict(&self, token: &str) -> Result<u32, RenderError> {
        self.token_to_id(token)
            .ok_or_else(|| RenderError::MissingSpecialToken(token.to_string()))
    }

    /// Encode `text` without adding model special tokens, returning the
    /// id sequence directly. Hot-path callers should batch text segments
    /// where possible, but per-segment encode is still significantly
    /// faster than the Python equivalent because there's no FFI hop.
    pub fn encode_no_special(&self, text: &str) -> Result<Encoded, RenderError> {
        let enc = self
            .inner
            .tok
            .encode_fast(text, false)
            .map_err(|e| RenderError::Tokenizer(e.to_string()))?;
        Ok(Encoded { enc })
    }

    /// Decode `ids` to text, including special tokens (matches the
    /// Python `tokenizer.decode(ids, skip_special_tokens=False)` used
    /// across the parsing layer).
    pub fn decode(&self, ids: &[u32]) -> Result<String, RenderError> {
        self.inner
            .tok
            .decode(ids, /*skip_special_tokens=*/ false)
            .map_err(|e| RenderError::Tokenizer(e.to_string()))
    }

    /// Borrow the underlying `tokenizers::Tokenizer` for advanced uses
    /// (batch encoding, vocab access, ...). Prefer the wrappers above on
    /// the hot path.
    pub fn raw(&self) -> &tokenizers::Tokenizer {
        &self.inner.tok
    }
}

/// Lightweight wrapper around `tokenizers::Encoding` exposing just the
/// id slice. Holding the encoding (instead of allocating a fresh
/// `Vec<u32>`) skips one copy on the way to `RenderBuf::ids`.
pub struct Encoded {
    enc: tokenizers::Encoding,
}

impl std::fmt::Debug for Encoded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoded")
            .field("len", &self.enc.len())
            .finish()
    }
}

impl Encoded {
    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        self.enc.get_ids()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.enc.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.enc.is_empty()
    }
}
