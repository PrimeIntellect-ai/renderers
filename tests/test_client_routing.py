"""CPU-only coverage for the routing transport JSON parser."""

import base64
import json
import struct

import pytest
from renderers.client import parse_generate_response


def _routing(dtype="uint8", *, with_weights=True):
    # Non-sorted expert slots and distinct exact FP32 coefficients catch any
    # stream swaps, slot changes, or accidental coefficient recomputation.
    ids = [7, 2, 0, 3, 1, 5, 2, 6]
    if dtype == "uint16":
        ids = [expert + 256 for expert in ids]
    id_bytes = struct.pack("<8B" if dtype == "uint8" else "<8H", *ids)
    weight_bytes = struct.pack(
        "<8f", 0.875, 0.125, 0.625, 0.375, 0.75, 0.25, 0.5625, 0.4375
    )
    routed = {
        "data": base64.b64encode(id_bytes).decode("ascii"),
        "shape": [2, 2, 2],
        "dtype": dtype,
        "start": 11,
        "source": {"name": "capture", "route_scale": 1.0},
    }
    if with_weights:
        routed.update(
            format_version=1,
            weights={
                "data": base64.b64encode(weight_bytes).decode("ascii"),
                "dtype": "float32",
                "coefficient_convention": "normalized_topk",
            },
        )
    return routed, id_bytes, weight_bytes


def _payload(routed):
    return {
        "request_id": "test-routing",
        "metadata": {"latency": 0.125},
        "choices": [{"index": 0, "token_ids": [7, 8], "routed_experts": routed}],
    }


def _plain_values(value):
    """Compare transport metadata without requiring a fast-path layout."""
    if isinstance(value, memoryview):
        return value.tobytes().decode("ascii")
    if isinstance(value, dict):
        return {key: _plain_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_values(item) for item in value]
    return value


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("with_weights", [False, True], ids=["legacy", "paired"])
def test_compact_routing_keeps_exact_raw_backed_streams(
    dtype, with_weights, monkeypatch
):
    routed, id_bytes, weight_bytes = _routing(dtype, with_weights=with_weights)
    expected = _payload(routed)
    raw = json.dumps(expected, separators=(",", ":")).encode()
    original_loads = json.loads
    decoded_inputs = []

    def record_loads(value, **kwargs):
        decoded_inputs.append(value)
        return original_loads(value, **kwargs)

    monkeypatch.setattr("renderers.client.json.loads", record_loads)
    parsed = parse_generate_response(raw)
    actual = parsed["choices"][0]["routed_experts"]
    streams = [(actual["data"], id_bytes)]
    if with_weights:
        streams.append((actual["weights"]["data"], weight_bytes))
    else:
        assert "weights" not in actual
        assert "format_version" not in actual
    assert len(decoded_inputs) == 1
    for encoded, expected_bytes in streams:
        assert isinstance(encoded, memoryview)
        assert encoded.obj is raw
        assert encoded.readonly
        assert encoded.tobytes() == base64.b64encode(expected_bytes)
        assert base64.b64decode(encoded, validate=True) == expected_bytes
        assert encoded.tobytes() not in decoded_inputs[0]
    assert _plain_values(parsed) == expected


@pytest.mark.parametrize("with_weights", [False, True], ids=["legacy", "paired"])
def test_compact_parser_does_not_copy_raw_chunks_before_join(with_weights):
    class UnslicedBytes(bytes):
        def __getitem__(self, key):
            if isinstance(key, slice):
                raise AssertionError("Use memoryviews instead of copying raw chunks")
            return super().__getitem__(key)

    routed, _, _ = _routing(with_weights=with_weights)
    expected = _payload(routed)
    raw = UnslicedBytes(json.dumps(expected, separators=(",", ":")).encode())

    parsed = parse_generate_response(raw)

    assert _plain_values(parsed) == expected
    actual = parsed["choices"][0]["routed_experts"]
    assert actual["data"].obj is raw
    if with_weights:
        assert actual["weights"]["data"].obj is raw


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("with_weights", [False, True], ids=["legacy", "paired"])
@pytest.mark.parametrize("layout", ["whitespace", "reordered", "weights_reordered"])
def test_json_layout_fallback_preserves_exact_streams(dtype, with_weights, layout):
    routed, id_bytes, weight_bytes = _routing(dtype, with_weights=with_weights)
    if layout == "reordered":
        routed = dict(reversed(list(routed.items())))
    elif layout == "weights_reordered" and with_weights:
        routed["weights"] = dict(reversed(list(routed["weights"].items())))
    expected = _payload(routed)
    kwargs = {"indent": 2} if layout == "whitespace" else {"separators": (",", ":")}
    raw = json.dumps(expected, **kwargs).encode()

    parsed = parse_generate_response(raw)

    actual = parsed["choices"][0]["routed_experts"]
    assert base64.b64decode(actual["data"], validate=True) == id_bytes
    if with_weights:
        assert (
            base64.b64decode(actual["weights"]["data"], validate=True) == weight_bytes
        )
    assert _plain_values(parsed) == expected


@pytest.mark.parametrize(
    "raw",
    [
        b'{"routed_experts":{"data":"T1RIRVI="},"choices":[{"routed_experts":{"data":"SUQ="}}]}',
        b'{"choices":[{}, {"routed_experts":{"data":"SUQ=","weights":{"data":"Vw=="}}}]}',
        b'{"weights":{"data":"T1RIRVI="},"choices":[{"routed_experts":{"data":"SUQ=","weights":{"data":"Vw=="}}}]}',
        b'{"choices":[{"routed_experts":{"data":"SUQ=","metadata":{"weights":{"data":"T1RIRVI="}},"weights":{"data":"Vw=="}}}]}',
        b'{"choices":[{"routed_experts":{"data":"SUQ=","weights":{"data":"Vw=="}}},{"routed_experts":{"data":"T1RIRVI=","weights":{"data":"V1JPTkc="}}}]}',
        b'{"choices":[{"routed_experts":{"data":"SUQ="},"weights":{"data":"T1RIRVI="}}]}',
        b'{"choices":[],"routed_experts":{"data":"SUQ="}}',
        b'[{"routed_experts":{"data":"SUQ="}}]',
    ],
    ids=[
        "unrelated-routing",
        "other-choice-only",
        "unrelated-weights-first",
        "nested-unrelated-weights",
        "multiple-choices",
        "weights-outside-routing",
        "empty-choices",
        "non-object-payload",
    ],
)
def test_does_not_attach_buffers_to_the_wrong_field(raw):
    assert _plain_values(parse_generate_response(raw)) == json.loads(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"choices":[{"routed_experts":{"data":"T1RIRVI=","data":"SUQ="}}]}',
        b'{"choices":[{"routed_experts":{"data":"SUQ=","weights":{"data":"T1RIRVI=","data":"Vw=="}}}]}',
        b'{"choices":[{"routed_experts":{"data":"T1RIRVI="},"routed_experts":{"data":"SUQ="}}]}',
        b'{"choices":[{"routed_experts":{"data":"SUQ=","weights":{"data":"T1RIRVI="},"weights":{"data":"Vw=="}}}]}',
        b'{"choices":[{"routed_experts":{"data":"T1RIRVI="}}],"choices":[{"routed_experts":{"data":"SUQ="}}]}',
        rb'{"choices":[{"routed_experts":{"data":"T1RIRVI=","d\u0061ta":"SUQ="}}]}',
    ],
    ids=["ids", "weights-data", "routing", "weights", "choices", "escaped-key"],
)
def test_duplicate_keys_keep_decoder_semantics_without_cross_wiring(raw):
    assert _plain_values(parse_generate_response(raw)) == json.loads(raw)


@pytest.mark.parametrize("stream", ["ids", "weights"])
@pytest.mark.parametrize("bad_bytes", [b"\x00", b"\n", b"\t", b"\xff", b"\\q"])
def test_fast_path_does_not_hide_invalid_json_inside_routing_data(stream, bad_bytes):
    routed, _, _ = _routing()
    raw = json.dumps(_payload(routed), separators=(",", ":")).encode()
    encoded = routed["data"] if stream == "ids" else routed["weights"]["data"]
    raw = raw.replace(encoded.encode(), bad_bytes)

    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        parse_generate_response(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"choices":[{"routed_experts":{"data":"AQ==',
        b'{"choices":[{"routed_experts":{"data":"AQ==","weights":{"data":"AA==',
        b'{"choices":[{"routed_experts":{"data":"AQ==",}}]}',
        b'{"choices":[{"routed_experts":{"data":"AQ=="}}]} trailing',
    ],
)
def test_truncated_or_malformed_response_fails_as_json(raw):
    with pytest.raises(json.JSONDecodeError):
        parse_generate_response(raw)


@pytest.mark.parametrize("stream", ["ids", "weights"])
def test_escaped_base64_is_unescaped_by_json_before_forwarding(stream):
    routed, id_bytes, weight_bytes = _routing()
    raw = json.dumps(_payload(routed), separators=(",", ":")).encode()
    encoded = routed["data"] if stream == "ids" else routed["weights"]["data"]
    escaped = f"\\u{ord(encoded[0]):04x}".encode() + encoded[1:].encode()
    raw = raw.replace(encoded.encode(), escaped)

    parsed = parse_generate_response(raw)

    actual = parsed["choices"][0]["routed_experts"]
    assert base64.b64decode(actual["data"], validate=True) == id_bytes
    assert base64.b64decode(actual["weights"]["data"], validate=True) == weight_bytes
    assert _plain_values(parsed) == _payload(routed)


def test_empty_streams_are_not_treated_as_missing():
    raw = b'{"choices":[{"routed_experts":{"data":"","weights":{"data":""}}}]}'
    routed = parse_generate_response(raw)["choices"][0]["routed_experts"]
    assert isinstance(routed["data"], memoryview)
    assert isinstance(routed["weights"]["data"], memoryview)
    assert routed["data"].tobytes() == routed["weights"]["data"].tobytes() == b""


@pytest.mark.parametrize("literal", [b"0.0e+0", b"0.0e+1", b'"0.0e+0"'])
def test_numeric_metadata_cannot_be_mistaken_for_a_stream(literal):
    raw = (
        b'{"metadata":'
        + literal
        + b',"choices":[{"routed_experts":{"data":"SUQ=","weights":{"data":"Vw=="}}}]}'
    )
    assert _plain_values(parse_generate_response(raw)) == json.loads(raw)
