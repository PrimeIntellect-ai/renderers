//! Helpers shared across renderers' `bridge_to_next_turn` implementations.

use crate::types::Message;

/// Returns `true` if any message in `new_messages` carries the assistant
/// role. Bridges refuse to retokenize assistant content because it would
/// replace model-sampled tokens with canonical template text, violating
/// the byte-for-byte contract.
#[inline]
pub fn reject_assistant_in_extension(new_messages: &[Message]) -> bool {
    new_messages.iter().any(|m| m.role == "assistant")
}

/// Return the longest prefix of `prev_prompt + prev_completion` that ends
/// at a turn-close token, or `None` if none exists and `synthesize_close`
/// is `None`.
///
/// Scans only within the completion segment — close tokens inside the
/// prompt are structural scaffolding, not turn boundaries the current
/// step produced.
///
/// Returns an owned `Vec<u32>` that the caller can mutate; the inputs are
/// borrowed.
pub fn trim_to_turn_close(
    previous_prompt_ids: &[u32],
    previous_completion_ids: &[u32],
    close_token_ids: &[u32],
    synthesize_close: Option<u32>,
) -> Option<Vec<u32>> {
    let prompt_len = previous_prompt_ids.len();
    let total_len = prompt_len + previous_completion_ids.len();

    // Walk the completion section backwards looking for a close token.
    for offset in (0..previous_completion_ids.len()).rev() {
        if close_token_ids.contains(&previous_completion_ids[offset]) {
            let mut out = Vec::with_capacity(prompt_len + offset + 1);
            out.extend_from_slice(previous_prompt_ids);
            out.extend_from_slice(&previous_completion_ids[..=offset]);
            return Some(out);
        }
    }

    let close = synthesize_close?;
    let mut out = Vec::with_capacity(total_len + 1);
    out.extend_from_slice(previous_prompt_ids);
    out.extend_from_slice(previous_completion_ids);
    out.push(close);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str) -> Message {
        Message {
            role: role.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn rejects_assistant_in_extension() {
        assert!(reject_assistant_in_extension(&[msg("assistant")]));
        assert!(!reject_assistant_in_extension(&[msg("user"), msg("tool")]));
        assert!(!reject_assistant_in_extension(&[]));
    }

    #[test]
    fn trim_to_close_keeps_prefix() {
        let prompt = vec![1, 2, 3];
        let completion = vec![4, 5, 9, 6, 9];
        let close = [9u32];
        let trimmed = trim_to_turn_close(&prompt, &completion, &close, None).unwrap();
        assert_eq!(trimmed, vec![1, 2, 3, 4, 5, 9, 6, 9]);
    }

    #[test]
    fn trim_to_close_finds_last_close() {
        let prompt = vec![1, 2];
        let completion = vec![9, 3, 4];
        let close = [9u32];
        let trimmed = trim_to_turn_close(&prompt, &completion, &close, None).unwrap();
        assert_eq!(trimmed, vec![1, 2, 9]);
    }

    #[test]
    fn trim_to_close_ignores_prompt_close() {
        let prompt = vec![9, 1, 2];
        let completion = vec![3, 4];
        let close = [9u32];
        assert!(trim_to_turn_close(&prompt, &completion, &close, None).is_none());
    }

    #[test]
    fn trim_to_close_synthesises_when_truncated() {
        let prompt = vec![1, 2];
        let completion = vec![3, 4];
        let close = [9u32];
        let trimmed = trim_to_turn_close(&prompt, &completion, &close, Some(9)).unwrap();
        assert_eq!(trimmed, vec![1, 2, 3, 4, 9]);
    }
}
