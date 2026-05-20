//! `<think>...</think>` retention rules shared across renderers.

use crate::types::Message;

/// Should `messages[msg_idx]`'s reasoning content be re-emitted even when
/// the chat template would normally drop it?
///
/// Returns `true` only as an override above the template default. Each
/// renderer ORs this into its own "render thinking?" condition.
///
/// Mirrors `renderers/base.py:should_preserve_past_thinking`.
pub fn should_preserve_past_thinking(
    messages: &[Message],
    msg_idx: usize,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,
) -> bool {
    if preserve_all_thinking {
        return true;
    }
    if !preserve_thinking_between_tool_calls {
        return false;
    }
    // Find the most recent user message (or None).
    let last_user: Option<usize> = messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(j, m)| (m.role == "user").then_some(j));

    let Some(last_user) = last_user else {
        // No user message before us: keep only if there's any tool turn
        // anywhere; rare path but matches the Python contract.
        return messages.iter().any(|m| m.role == "tool");
    };

    if msg_idx <= last_user {
        return false;
    }
    // The current segment must contain a tool response for the block to
    // count as an in-flight tool cycle.
    messages[last_user + 1..]
        .iter()
        .any(|m| m.role == "tool")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(role: &str) -> Message {
        Message {
            role: role.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn preserve_all_wins() {
        let msgs = vec![m("user"), m("assistant")];
        assert!(should_preserve_past_thinking(&msgs, 1, true, false));
    }

    #[test]
    fn between_tool_calls_keeps_active_cycle() {
        // Tool-cycle assistants (after the last user) are kept; the
        // current tool block must contain at least one `tool` turn.
        let msgs = vec![m("user"), m("assistant"), m("tool"), m("assistant")];
        // both assistants are after last_user=0 and the segment has a tool
        assert!(should_preserve_past_thinking(&msgs, 1, false, true));
        assert!(should_preserve_past_thinking(&msgs, 3, false, true));
        // a prior tool cycle (before a later user) is dropped
        let msgs2 = vec![
            m("user"),
            m("assistant"),
            m("tool"),
            m("assistant"),
            m("user"),
            m("assistant"),
        ];
        // assistant at idx=1 is before last_user=4 → dropped
        assert!(!should_preserve_past_thinking(&msgs2, 1, false, true));
        // assistant at idx=5 is after last_user=4 but segment has no tool → dropped
        assert!(!should_preserve_past_thinking(&msgs2, 5, false, true));
    }

    #[test]
    fn between_tool_calls_drops_without_tool() {
        let msgs = vec![m("user"), m("assistant"), m("assistant")];
        assert!(!should_preserve_past_thinking(&msgs, 2, false, true));
    }
}
