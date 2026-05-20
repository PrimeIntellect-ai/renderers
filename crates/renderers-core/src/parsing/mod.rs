//! Token-level parsing primitives shared across family-specific parsers.
//!
//! The strategy matches the Python implementation: scan token ids for
//! special-token boundaries (no decoded-text regex on the full stream),
//! then decode only inside the bounded segments. This is the only way to
//! avoid false positives from content that happens to look like a
//! special token.
//!
//! All helpers operate on `&[u32]` slices and are `#[inline]`-marked so
//! they vanish into the family parsers at -O.

pub mod deepseek_v3;
pub mod glm;
pub mod kimi_k2;
pub mod minimax;
pub mod qwen3;
pub mod qwen35;

use crate::tokenizer::Tokenizer;
use crate::types::RenderError;

/// Find the first index of `target` in `ids`, or `None`.
#[inline]
pub fn find(ids: &[u32], target: u32) -> Option<usize> {
    ids.iter().position(|&x| x == target)
}

/// Find the first index of `target` in `ids[start..]`, or `None`.
#[inline]
pub fn find_from(ids: &[u32], target: u32, start: usize) -> Option<usize> {
    ids[start..].iter().position(|&x| x == target).map(|i| i + start)
}

/// Find the first index of any token in `targets`, or `None`. `targets`
/// is small (≤ a few) for every renderer, so a linear contains-check is
/// faster than a `HashSet`.
#[inline]
pub fn find_any(ids: &[u32], targets: &[u32]) -> Option<usize> {
    ids.iter().position(|x| targets.contains(x))
}

/// Truncate `ids` at the first stop token. Returns the prefix as a
/// borrowed slice — no allocation.
#[inline]
pub fn strip_stop_tokens<'a>(ids: &'a [u32], stop_ids: &[u32]) -> &'a [u32] {
    match find_any(ids, stop_ids) {
        Some(i) => &ids[..i],
        None => ids,
    }
}

/// Decode `ids` via `tokenizer.decode(ids, skip_special_tokens=False)`.
/// Returns an empty string for empty input without calling the
/// tokenizer (saves an FFI-free but still measurable ~µs per call).
#[inline]
pub fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String, RenderError> {
    if ids.is_empty() {
        return Ok(String::new());
    }
    tokenizer.decode(ids)
}
