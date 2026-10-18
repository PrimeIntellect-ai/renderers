"""Live responses derived from the independent template/encoder oracle.

History serialization is not always a live completion: R1 drops reasoning,
Harmony drops final-turn analysis, and Gemma drops empty thinking prefills.
These explicit adapters describe those reference contracts. Never repair a
prefix mismatch by silently slicing at a common prefix or asking our renderer
to supply the expected output.
"""

import json
import re

from reference_rendering import _harmony_messages, render_reference

REASONING = "Check the weather carefully."
ANSWER = "It is sunny in Tokyo."


def reference_continuation(tok, case, kwargs, messages, tools, *, calls=0):
    options = {
        k: v for k, v in (dict(case.oracle_defaults) | kwargs).items() if v is not None
    }

    def render(conversation, generation=False, **overrides):
        return render_reference(
            tok,
            conversation,
            oracle=case.oracle,
            tools=tools,
            add_generation_prompt=generation,
            **(options | overrides),
        )

    prompt = render(messages, generation=True)
    prompt_text = tok.decode(prompt, skip_special_tokens=False)
    # Resolve the mode from a reference user-turn prefix even when a tool
    # observation continues the same assistant turn without repeating it.
    first_user = next(
        i for i, message in enumerate(messages) if message["role"] == "user"
    )
    mode_prefix = tok.decode(render(messages[: first_user + 1], generation=True))
    disabled = bool(
        re.search(r"(?:</think>|</think:opensource>|<channel\|>)\s*$", mode_prefix)
    )
    assistant = {"role": "assistant", "content": "" if calls else ANSWER}
    if not disabled:
        assistant["reasoning_content"] = REASONING
    if calls:
        assistant["tool_calls"] = [
            {
                "id": f"functions.get_weather:{index}",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": city}},
            }
            for index, city in enumerate(["Tokyo", "Berlin"][:calls])
        ]
        if case.resolved_renderer in {"deepseek-v3", "deepseek-r1"}:
            # These upstream templates require null content and JSON strings.
            assistant["content"] = None
            for call in assistant["tool_calls"]:
                fn = call["function"]
                fn["arguments"] = json.dumps(fn["arguments"])

    # Raw-history modes expect an already serialized assistant body. Serialize
    # our structured fixture through the reference's normal branch; these flags
    # do not change the live generation prefix (checked independently below).
    serialization = {
        key: False
        for key in ("raw_last_assistant", "render_assistant_messages_raw")
        if options.get(key)
    }
    assert render(messages, generation=True, **serialization) == prompt
    full = render(messages + [assistant], **serialization)
    full_text = tok.decode(full, skip_special_tokens=False)

    if case.oracle == "harmony" and not calls:
        from openai_harmony import (
            Conversation,
            HarmonyEncodingName,
            Message,
            Role,
            RenderConversationConfig,
            load_harmony_encoding,
        )

        encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        history = _harmony_messages(
            messages,
            tools=tools,
            reasoning_effort=options.get("reasoning_effort", "medium"),
            conversation_start_date=options.get("conversation_start_date"),
        )
        history += [
            Message.from_role_and_content(Role.ASSISTANT, REASONING).with_channel(
                "analysis"
            ),
            Message.from_role_and_content(Role.ASSISTANT, ANSWER).with_channel("final"),
        ]
        full = encoder.render_conversation_for_training(
            Conversation.from_messages(history),
            RenderConversationConfig(auto_drop_analysis=False),
        )
    elif case.resolved_renderer == "deepseek-r1" and prompt_text.endswith("<think>\n"):
        # R1's generation prompt opens thinking, but its history branch removes
        # it. Restore the live reasoning segment before the template's answer.
        prefill = "<think>\n"
        assert prompt_text.endswith(prefill)
        history_prefix = prompt_text[: -len(prefill)]
        assert full_text.startswith(history_prefix)
        full_text = (
            prompt_text + REASONING + "</think>" + full_text[len(history_prefix) :]
        )
        full = tok.encode(full_text, add_special_tokens=False)
    elif case.resolved_renderer == "gemma4" and prompt_text.endswith(
        "<|channel>thought\n<channel|>"
    ):
        prefill = "<|channel>thought\n<channel|>"
        assert prompt_text.endswith(prefill)
        history_prefix = prompt_text[: -len(prefill)]
        assert full_text.startswith(history_prefix)
        full_text = prompt_text + full_text[len(history_prefix) :]
        full = tok.encode(full_text, add_special_tokens=False)
    elif (
        case.resolved_renderer == "laguna-s-2.1"
        and disabled
        and options.get("preserve_thinking")
    ):
        # History preserves an empty opener; generation supplies only the close.
        history_prefix = prompt_text.removesuffix("</think>") + "<think></think>"
        assert full_text.startswith(history_prefix)
        full_text = prompt_text + full_text[len(history_prefix) :]
        full = tok.encode(full_text, add_special_tokens=False)

    if (
        case.resolved_renderer in {"kimi-k2.5", "kimi-k2.6"}
        and messages[-1]["role"] == "tool"
    ):
        # Kimi's final-answer history branch strips the preceding tool turn's
        # reasoning as well as the final turn's reasoning. Extract only the
        # last assistant body and preserve the actual sampling prompt.
        header = "<|im_assistant|>assistant<|im_middle|>"
        body = tok.decode(full).rsplit(header, 1)[1]
        if not disabled:
            assert prompt_text.endswith(header + "<think>")
            assert body.startswith("<think></think>")
            body = REASONING + body[len("<think>") :]
        else:
            assert prompt_text.endswith(header + "</think>")
            assert body.startswith("</think>")
            body = body[len("</think>") :]
        full = tok.encode(prompt_text + body, add_special_tokens=False)

    assert full[: len(prompt)] == prompt, (
        "Reference history is not a live continuation; add an explicit oracle adapter",
        case.model,
        kwargs,
        prompt_text[-200:],
        tok.decode(full)[-300:],
    )
    completion = full[len(prompt) :]
    # Expectations come from the reference, never from parse_response. Some
    # non-reasoning templates intentionally ignore reasoning_content inputs.
    reasoning = REASONING if REASONING in tok.decode(completion) else ""
    return prompt, completion, reasoning
