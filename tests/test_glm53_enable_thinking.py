"""GLM-5.3 ``enable_thinking=False`` — thinking-off mode.

The official GLM-5.3 chat template exposes no thinking-off kwarg, so this
mode is a deliberate, documented departure from it (see ``GLM53Renderer``):
with ``enable_thinking=False`` the generation prompt prefills the empty
think block, and the block's closing token on assistant turns is scaffold
instead of a trained target. Inference must render the same prompt.
"""

from pydantic import TypeAdapter

from renderers import create_renderer
from renderers.base import build_training_sample, load_tokenizer
from renderers.configs import GLM53RendererConfig, RendererConfig

# Token spellings are assembled from parts; never print the assembled values.
_T = chr(60) + "think" + chr(62)
_T_END = chr(60) + "/think" + chr(62)

MODEL = "zai-org/GLM-5.3-BF16"
MSGS = [
    {"role": "user", "content": "Q? Answer:"},
    {"role": "assistant", "content": "B"},
]


def _renderer(enable_thinking: bool):
    return create_renderer(
        load_tokenizer(MODEL), GLM53RendererConfig(enable_thinking=enable_thinking)
    )


def test_default_behavior_unchanged():
    renderer = _renderer(True)
    gen = renderer.render(MSGS[:1], add_generation_prompt=True)
    assert gen.token_ids[-2:] == [renderer._assistant, renderer._think]
    assert all(m is False for m in gen.sampled_mask[-2:])

    sample = build_training_sample(renderer, MSGS, ensure_final_stop=True)
    trained = [t for t, m in zip(sample.token_ids, sample.loss_mask) if m]
    assert trained == [renderer._think_end] + renderer._encode("B") + [
        renderer._endoftext
    ]
    rendered = renderer.render(MSGS)
    pos = rendered.token_ids.index(renderer._think_end)
    assert rendered.sampled_mask[pos] is True
    assert rendered.is_content[pos] is True


def test_thinking_off_generation_prompt_prefills_empty_block():
    renderer = _renderer(False)
    gen = renderer.render(MSGS[:1], add_generation_prompt=True)
    assert gen.token_ids[-3:] == [
        renderer._assistant,
        renderer._think,
        renderer._think_end,
    ]
    assert all(m is False for m in gen.sampled_mask[-3:])


def test_thinking_off_trains_only_answer_and_stop():
    """The acceptance case: SFT on [user, assistant "B"] trains [B, eos]."""
    renderer = _renderer(False)
    sample = build_training_sample(renderer, MSGS, ensure_final_stop=True)
    trained = [t for t, m in zip(sample.token_ids, sample.loss_mask) if m]
    assert trained == renderer._encode("B") + [renderer._endoftext]

    rendered = renderer.render(MSGS)
    pos = rendered.token_ids.index(renderer._think_end)
    assert rendered.sampled_mask[pos] is False
    assert rendered.is_content[pos] is False


def test_thinking_off_covers_historical_assistant_turns():
    """Both assistant turns keep the empty block untrained, so multi-turn
    thinking-off data never teaches the model to close the think block.
    Their sampled content still trains."""
    renderer = _renderer(False)
    msgs = [
        {"role": "user", "content": "Q1?"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2?"},
        {"role": "assistant", "content": "B"},
    ]
    sample = build_training_sample(renderer, msgs, ensure_final_stop=True)
    trained = [t for t, m in zip(sample.token_ids, sample.loss_mask) if m]
    assert trained == (
        renderer._encode("A1")
        + [renderer._user]
        + renderer._encode("B")
        + [renderer._endoftext]
    )
    rendered = renderer.render(msgs)
    closes = [i for i, t in enumerate(rendered.token_ids) if t == renderer._think_end]
    assert len(closes) == 2
    assert all(rendered.sampled_mask[p] is False for p in closes)
    assert all(rendered.is_content[p] is False for p in closes)


def test_thinking_off_keeps_reasoning_turns_template_faithful():
    """Reasoning-bearing turns were not sampled under a thinking-off config;
    their block stays template-faithful (sampled + content)."""
    renderer = _renderer(False)
    msgs = [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "B", "reasoning_content": "because"},
    ]
    rendered = renderer.render(msgs)
    pos = rendered.token_ids.index(renderer._think_end)
    assert rendered.sampled_mask[pos] is True
    assert rendered.is_content[pos] is True


def test_union_config_accepts_enable_thinking():
    parsed = TypeAdapter(RendererConfig).validate_python(
        {"name": "glm-5.3", "enable_thinking": False}
    )
    assert isinstance(parsed, GLM53RendererConfig)
    assert parsed.enable_thinking is False
    assert GLM53RendererConfig().enable_thinking is True


def test_thinking_off_bridge_matches_generation_prompt():
    """``bridge_to_next_turn`` extends with the same prefilled prompt that
    ``render(add_generation_prompt=True)`` produces."""
    renderer = _renderer(False)
    prompt_ids = renderer.render_ids(MSGS[:1], add_generation_prompt=True)
    completion = renderer._encode("B") + [renderer._endoftext]
    bridge = renderer.bridge_to_next_turn(
        prompt_ids, completion, [{"role": "user", "content": "Again?"}]
    )
    assert bridge is not None
    expected = (
        list(prompt_ids)
        + completion
        + [renderer._user]
        + renderer._encode("Again?")
        + [renderer._assistant, renderer._think, renderer._think_end]
    )
    assert bridge.token_ids == expected
    assert all(m is False for m in bridge.sampled_mask)


def test_default_mode_masks_unchanged_on_empty_think_turns():
    """With the default ``enable_thinking=True``, token ids and masks are
    unchanged by this feature: every assistant turn's closing think token
    stays a trained target (``is_sampled=True, is_content=True``), even for
    historical turns whose think block is empty."""
    renderer = _renderer(True)
    msgs = [
        {"role": "user", "content": "Q1?"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2?"},
        {"role": "assistant", "content": "B"},
    ]
    for kw, n_opens in (({}, 2), ({"add_generation_prompt": True}, 3)):
        rendered = renderer.render(msgs, **kw)
        closes = [
            i for i, t in enumerate(rendered.token_ids) if t == renderer._think_end
        ]
        opens = [i for i, t in enumerate(rendered.token_ids) if t == renderer._think]
        assert len(closes) == 2 and len(opens) == n_opens
        assert all(rendered.sampled_mask[p] is True for p in closes)
        assert all(rendered.is_content[p] is True for p in closes)
        assert all(rendered.sampled_mask[p] is False for p in opens)
