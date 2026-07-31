"""Tests for the rlm renderer (RLM chat format).

The reference tokenizer is ``PrimeIntellect/RLM-Chat-Template`` (private):
the Nemotron-3 Super tokenizer with reserved ``<SPECIAL_18>``..``<SPECIAL_27>``
slots renamed to the ten single-token role tags, plus the minimal chat
template. The fixture below reconstructs it from the public Nemotron-3 Super
tokenizer so CI needs no private-hub access; the byte-level rename recipe is
identical to the published repo's build.
"""

import json
from pathlib import Path

import pytest
from renderers import create_renderer
from renderers.base import ToolCallParseStatus, load_tokenizer
from renderers.configs import RlmRendererConfig, config_from_name

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

TAGS = {
    18: "<system>",
    19: "</system>",
    20: "<user>",
    21: "</user>",
    22: "<assistant>",
    23: "</assistant>",
    24: "<ipython>",
    25: "</ipython>",
    26: "<output>",
    27: "</output>",
}

# Mirrors PrimeIntellect/RLM-Chat-Template chat_template.jinja exactly.
CHAT_TEMPLATE = (
    "{%- if tools -%}"
    '{{- raise_exception("This template does not support tools=. The single built-in tool is ipython, invoked inline as <ipython>...</ipython> in assistant turns.") -}}'
    "{%- endif -%}"
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'system' -%}"
    "{{- '<system>' + message['content'] + '</system>' -}}"
    "{%- elif message['role'] == 'user' -%}"
    "{{- '<user>' + message['content'] + '</user>' -}}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{{- '<assistant>' -}}"
    "{%- if message['content'] -%}{{- message['content'] -}}{%- endif -%}"
    "{%- if message['tool_calls'] is defined and message['tool_calls'] -%}"
    "{%- set call = message['tool_calls'][0]['function'] -%}"
    "{{- '<ipython>' + call['arguments']['code'] + '</ipython>' -}}"
    "{%- endif -%}"
    "{{- '</assistant>' -}}"
    "{%- elif message['role'] == 'tool' -%}"
    "{{- '<output>' + message['content'] + '</output>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<assistant>' -}}{%- endif -%}"
)


@pytest.fixture(scope="module")
def rlm_tokenizer(tmp_path_factory):
    """Rebuild the RLM-Chat-Template tokenizer from the public base model."""
    from huggingface_hub import hf_hub_download

    out = tmp_path_factory.mktemp("rlm-tokenizer")
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = Path(hf_hub_download(BASE_MODEL, fname))
        data = json.loads(src.read_text())
        if fname == "tokenizer.json":
            for tok in data["added_tokens"]:
                if tok["id"] in TAGS:
                    tok["content"] = TAGS[tok["id"]]
            vocab = data["model"]["vocab"]
            for i, tag in TAGS.items():
                del vocab[f"<SPECIAL_{i}>"]
                vocab[tag] = i
        elif fname == "tokenizer_config.json":
            for i, tag in TAGS.items():
                data["added_tokens_decoder"][str(i)]["content"] = tag
            data["eos_token"] = "</assistant>"
            data.pop("chat_template", None)
        else:
            data["eos_token"]["content"] = "</assistant>"
        (out / fname).write_text(json.dumps(data, ensure_ascii=False))
    (out / "chat_template.jinja").write_text(CHAT_TEMPLATE)
    return load_tokenizer(str(out))


@pytest.fixture(scope="module")
def renderer(rlm_tokenizer):
    return create_renderer(rlm_tokenizer, RlmRendererConfig())


MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Fix the bug in foo.py"},
    {
        "role": "assistant",
        "content": "<think>Look at the file first.</think>I'll inspect foo.py.",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "ipython",
                    "arguments": {"code": "print(open('foo.py').read())"},
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "def foo():\n    return 1/0\n"},
    {"role": "assistant", "content": "<think>Division by zero.</think>Fixed."},
]


def test_tags_are_single_tokens(rlm_tokenizer):
    for i, tag in TAGS.items():
        assert rlm_tokenizer.encode(tag, add_special_tokens=False) == [i]
    assert rlm_tokenizer.eos_token_id == 23


@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_render_parity_with_chat_template(renderer, rlm_tokenizer, add_generation_prompt):
    for upto in (2, 3, 4, 5):
        msgs = MESSAGES[:upto]
        if add_generation_prompt and msgs[-1]["role"] == "assistant":
            continue
        expected = rlm_tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_generation_prompt)
        if not isinstance(expected, list):
            expected = expected["input_ids"]
        got = renderer.render_ids(msgs, add_generation_prompt=add_generation_prompt)
        assert got == expected, f"parity mismatch at upto={upto}"


def test_thinking_never_dropped(renderer, rlm_tokenizer):
    ids = renderer.render_ids(MESSAGES)
    text = rlm_tokenizer.decode(ids)
    assert "<think>Look at the file first.</think>" in text
    assert "<think>Division by zero.</think>" in text


def test_reasoning_content_is_restored_not_dropped(renderer, rlm_tokenizer):
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "answer", "reasoning_content": "hidden plan"},
    ]
    text = rlm_tokenizer.decode(renderer.render_ids(msgs))
    assert "<think>hidden plan</think>answer" in text


def test_tools_validation(renderer):
    ipython_tool = {
        "type": "function",
        "function": {"name": "ipython", "parameters": {"type": "object"}},
    }
    msgs = MESSAGES[:2]
    # exactly [ipython]: accepted, renders identically to no tools
    assert renderer.render_ids(msgs, tools=[ipython_tool]) == renderer.render_ids(msgs)
    with pytest.raises(ValueError, match="only tool is ipython"):
        renderer.render_ids(msgs, tools=[{"type": "function", "function": {"name": "bash"}}])
    with pytest.raises(ValueError, match="exactly one tool"):
        renderer.render_ids(msgs, tools=[ipython_tool, ipython_tool])


def test_render_rejects_bad_tool_calls(renderer):
    base = {"role": "assistant", "content": ""}
    with pytest.raises(ValueError, match="unknown tool"):
        renderer.render_ids(
            [
                {"role": "user", "content": "u"},
                {
                    **base,
                    "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"code": "x"}}}],
                },
            ]
        )
    with pytest.raises(ValueError, match="exactly one ipython call"):
        tc = {"type": "function", "function": {"name": "ipython", "arguments": {"code": "x"}}}
        renderer.render_ids([{"role": "user", "content": "u"}, {**base, "tool_calls": [tc, tc]}])


def test_render_accepts_trace_shape_tool_calls(renderer, rlm_tokenizer):
    # HF datasets store verifiers-trace tool calls as flat JSON strings.
    msgs = [
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": "c",
            "tool_calls": ['{"id": "t1", "name": "ipython", "arguments": "{\\"code\\": \\"print(1)\\"}"}'],
        },
    ]
    text = rlm_tokenizer.decode(renderer.render_ids(msgs))
    assert "<ipython>print(1)</ipython>" in text


def test_parse_response_roundtrip(renderer, rlm_tokenizer):
    body = "<think>plan</think>text<ipython>print(1)</ipython>"
    ids = rlm_tokenizer.encode(body, add_special_tokens=False) + [23]
    parsed = renderer.parse_response(ids)
    assert parsed.content == "<think>plan</think>text"
    assert parsed.reasoning_content is None
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.status == ToolCallParseStatus.OK
    assert call.name == "ipython"
    assert call.arguments == {"code": "print(1)"}
    start, end = call.token_span
    stripped = ids[:-1]
    assert stripped[start] == 24 and stripped[end - 1] == 25


def test_parse_response_content_only(renderer, rlm_tokenizer):
    ids = rlm_tokenizer.encode("just an answer", add_special_tokens=False) + [23]
    parsed = renderer.parse_response(ids)
    assert parsed.content == "just an answer"
    assert parsed.tool_calls == []


def test_parse_response_unclosed_ipython(renderer, rlm_tokenizer):
    ids = rlm_tokenizer.encode("x<ipython>print(", add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert parsed.content == "x"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].status == ToolCallParseStatus.UNCLOSED_BLOCK


def test_stop_tokens(renderer):
    assert renderer.get_stop_token_ids() == [23]


def test_bridge_extends_exactly(renderer):
    prompt = renderer.render_ids(MESSAGES[:2], add_generation_prompt=True)
    # completion the model would sample: body + </assistant>
    full_turn = renderer.render(MESSAGES[:3])
    completion = full_turn.token_ids[len(prompt) :]
    bridged = renderer.bridge_to_next_turn(prompt, completion, [MESSAGES[3]])
    assert bridged is not None
    expected = renderer.render_ids(MESSAGES[:4], add_generation_prompt=True)
    assert bridged.token_ids == expected
    assert bridged.token_ids[: len(prompt) + len(completion)] == prompt + completion


def test_bridge_rejects_assistant_messages(renderer):
    prompt = renderer.render_ids(MESSAGES[:2], add_generation_prompt=True)
    out = renderer.bridge_to_next_turn(prompt, [23], [{"role": "assistant", "content": "no"}])
    assert out is None


def test_config_rejects_non_all_retention():
    with pytest.raises(Exception, match="never drops thinking"):
        RlmRendererConfig(thinking_retention="tool_cycle")
    assert config_from_name("rlm").thinking_retention is None


def test_content_mask_roles(renderer):
    rendered = renderer.render(MESSAGES)
    spans = rendered.content_token_spans_by_role()
    assert set(spans) >= {"assistant", "tool", "user", "system"}
    # assistant invariant: is_content == sampled_mask
    for is_c, is_s, idx in zip(rendered.is_content, rendered.sampled_mask, rendered.message_indices):
        if idx >= 0 and MESSAGES[idx]["role"] == "assistant":
            assert is_c == is_s
