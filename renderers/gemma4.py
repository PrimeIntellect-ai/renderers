"""Gemma 4 renderer with deterministic text, tool, thinking, and image support.

The implementation mirrors Google's canonical 2026-07-09 chat template and
``Gemma4Processor`` image expansion.  A template ``<|image|>`` marker becomes
``<|image>`` + N x ``<|image|>`` + ``<image|>``, where N is the dynamic soft
token count returned by ``Gemma4ImageProcessor`` for that image's aspect ratio.

Gemma 4 uses a continuation-style tool loop: the model ends a tool call by
sampling ``<|tool_response>``; the runtime appends one or more response blocks,
and generation resumes in the same model turn.  The bridge implementation
preserves that boundary without re-rendering sampled history.

The 26B/31B disabled-thinking template revision prefills an empty thought
channel in every generation prompt. Historical assistant turns without
reasoning re-emit that exact wrapper so a grown full render preserves the byte
prefix under which each completion was sampled. This is a deliberate stability
deviation from ``apply_chat_template``; E2B/E4B remain template-faithful because
their revision has no such prefill.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from renderers.base import (
    Message,
    MultiModalData,
    ParsedResponse,
    ParsedToolCall,
    PlaceholderRange,
    RenderedTokens,
    ToolCallParseStatus,
    ToolSpec,
    Tokenizer,
    _content_mask_or_empty,
    _require_transformers,
    attribute_text_segments,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import Gemma4RendererConfig
from renderers.qwen3_vl import (
    _image_hash,
    _is_image_part,
    _is_video_part,
    _load_pil_image,
)

_ESCAPE = '<|"|>'
_EMPTY_THOUGHT_PREFILL_MODELS = {
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
}


class _Emitter:
    """BPE-safe token emitter with per-token attribution side channels."""

    def __init__(self, encode_fn, tokenizer=None, msg_idx: int = -1):
        self._encode = encode_fn
        self._tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.message_indices: list[int] = []
        self.sampled: list[bool] = []
        self.is_content: list[bool] = []
        self._segments: list[tuple[str, bool]] = []
        self._buf_idx = msg_idx
        self._buf_sampled = False
        self.msg_idx = msg_idx

    def set_msg_idx(self, msg_idx: int) -> None:
        if self._segments:
            self._flush()
        self.msg_idx = msg_idx
        self._buf_idx = msg_idx

    def text(self, text: str, *, is_sampled: bool, is_content: bool) -> None:
        if not text:
            return
        if self._segments and (
            self._buf_idx != self.msg_idx or self._buf_sampled != is_sampled
        ):
            self._flush()
        if not self._segments:
            self._buf_idx = self.msg_idx
            self._buf_sampled = is_sampled
        self._segments.append((text, is_content))

    def special(self, token_id: int, *, is_sampled: bool, is_content: bool) -> None:
        if self._segments:
            self._flush()
        self.token_ids.append(token_id)
        self.message_indices.append(self.msg_idx)
        self.sampled.append(is_sampled)
        self.is_content.append(is_content)

    def cursor(self) -> int:
        if self._segments:
            self._flush()
        return len(self.token_ids)

    def finalize(self) -> None:
        if self._segments:
            self._flush()

    def _flush(self) -> None:
        segments = self._segments
        self._segments = []
        if not segments:
            return
        first_content = segments[0][1]
        if all(is_content == first_content for _, is_content in segments):
            ids = self._encode("".join(text for text, _ in segments))
            self.token_ids.extend(ids)
            self.message_indices.extend([self._buf_idx] * len(ids))
            self.sampled.extend([self._buf_sampled] * len(ids))
            self.is_content.extend([first_content] * len(ids))
            return
        assert self._tokenizer is not None, (
            "Gemma4 _Emitter needs a tokenizer for mixed body/scaffold text."
        )
        for token_id, is_content in attribute_text_segments(self._tokenizer, segments):
            self.token_ids.append(token_id)
            self.message_indices.append(self._buf_idx)
            self.sampled.append(self._buf_sampled)
            self.is_content.append(is_content)


def _dictsort(value: Mapping[str, Any]):
    """Match Jinja's default case-insensitive ``dictsort`` ordering."""
    return sorted(value.items(), key=lambda item: str(item[0]).lower())


def _format_argument(argument: Any, *, escape_keys: bool = True) -> str:
    """Mirror Gemma 4's JSON-like argument serializer."""
    if argument is None:
        return "null"
    if isinstance(argument, str):
        return f"{_ESCAPE}{argument}{_ESCAPE}"
    if isinstance(argument, bool):
        return "true" if argument else "false"
    if isinstance(argument, Mapping):
        fields = []
        for key, value in _dictsort(argument):
            rendered_key = f"{_ESCAPE}{key}{_ESCAPE}" if escape_keys else str(key)
            fields.append(
                f"{rendered_key}:{_format_argument(value, escape_keys=escape_keys)}"
            )
        return "{" + ",".join(fields) + "}"
    if isinstance(argument, Sequence) and not isinstance(
        argument, (str, bytes, bytearray)
    ):
        return (
            "["
            + ",".join(
                _format_argument(item, escape_keys=escape_keys) for item in argument
            )
            + "]"
        )
    return str(argument)


def _format_parameters(
    properties: Mapping[str, Any],
    required: Sequence[str] | None,
    *,
    filter_keys: bool = False,
) -> str:
    standard_keys = {"description", "type", "properties", "required", "nullable"}
    rendered: list[str] = []
    for key, raw_value in _dictsort(properties):
        if filter_keys and key in standard_keys:
            continue
        value = raw_value if isinstance(raw_value, Mapping) else {}
        fields: list[str] = []
        description = value.get("description")
        if description:
            fields.append(f"description:{_ESCAPE}{description}{_ESCAPE}")
        value_type = str(value.get("type") or "").upper()
        if value_type == "STRING" and value.get("enum"):
            fields.append(f"enum:{_format_argument(value['enum'])}")
        elif value_type == "ARRAY":
            items = value.get("items")
            if isinstance(items, Mapping) and items:
                item_fields: list[str] = []
                for item_key, item_value in _dictsort(items):
                    if item_value is None:
                        continue
                    if item_key == "properties":
                        nested = (
                            _format_parameters(
                                item_value,
                                items.get("required") or [],
                            )
                            if isinstance(item_value, Mapping)
                            else ""
                        )
                        item_fields.append(f"properties:{{{nested}}}")
                    elif item_key == "required":
                        vals = ",".join(f"{_ESCAPE}{v}{_ESCAPE}" for v in item_value)
                        item_fields.append(f"required:[{vals}]")
                    elif item_key == "type":
                        upper = (
                            item_value.upper()
                            if isinstance(item_value, str)
                            else [str(v).upper() for v in item_value]
                        )
                        item_fields.append(f"type:{_format_argument(upper)}")
                    else:
                        item_fields.append(f"{item_key}:{_format_argument(item_value)}")
                fields.append("items:{" + ",".join(item_fields) + "}")
        if value.get("nullable"):
            fields.append("nullable:true")
        if value_type == "OBJECT":
            nested_properties = value.get("properties")
            if isinstance(nested_properties, Mapping):
                nested = _format_parameters(
                    nested_properties,
                    value.get("required") or [],
                )
                fields.append(f"properties:{{{nested}}}")
            elif isinstance(value, Mapping):
                nested = _format_parameters(
                    value,
                    value.get("required") or [],
                    filter_keys=True,
                )
                fields.append(f"properties:{{{nested}}}")
            if value.get("required"):
                vals = ",".join(
                    f"{_ESCAPE}{item}{_ESCAPE}" for item in value["required"]
                )
                fields.append(f"required:[{vals}]")
        fields.append(f"type:{_ESCAPE}{value_type}{_ESCAPE}")
        rendered.append(f"{key}:{{{','.join(fields)}}}")
    return ",".join(rendered)


def _unwrap_tool(tool: ToolSpec) -> Mapping[str, Any]:
    function = tool.get("function") if isinstance(tool, Mapping) else None
    if isinstance(function, Mapping):
        return function
    return tool


def _format_function_declaration(tool: ToolSpec) -> str:
    function = _unwrap_tool(tool)
    name = function.get("name") or ""
    description = function.get("description") or ""
    out = f"declaration:{name}{{description:{_ESCAPE}{description}{_ESCAPE}"
    params = function.get("parameters")
    if isinstance(params, Mapping) and params:
        param_fields: list[str] = []
        properties = params.get("properties")
        if isinstance(properties, Mapping) and properties:
            param_fields.append(
                "properties:{"
                + _format_parameters(properties, params.get("required") or [])
                + "}"
            )
        required = params.get("required")
        if required:
            vals = ",".join(f"{_ESCAPE}{item}{_ESCAPE}" for item in required)
            param_fields.append(f"required:[{vals}]")
        if params.get("type"):
            param_fields.append(f"type:{_ESCAPE}{str(params['type']).upper()}{_ESCAPE}")
        out += ",parameters:{" + ",".join(param_fields) + "}"
    if "response" in function:
        response = function.get("response")
        response = response if isinstance(response, Mapping) else {}
        response_fields: list[str] = []
        if response.get("description"):
            response_fields.append(
                f"description:{_ESCAPE}{response['description']}{_ESCAPE}"
            )
        if str(response.get("type") or "").upper() == "OBJECT":
            response_fields.append(f"type:{_ESCAPE}OBJECT{_ESCAPE}")
        out += ",response:{" + ",".join(response_fields) + "}"
    return out + "}"


def _strip_thinking(text: str) -> str:
    result = ""
    for part in text.split("<channel|>"):
        if "<|channel>" in part:
            result += part.split("<|channel>", 1)[0]
        else:
            result += part
    return result.strip()


def _coerce_tool_arguments(arguments: Any) -> Mapping[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise TypeError(
                "Gemma 4 tool_calls[].function.arguments must be a JSON object."
            ) from exc
        if isinstance(parsed, Mapping):
            return parsed
    raise TypeError("Gemma 4 tool_calls[].function.arguments must be a JSON object.")


class _ArgumentParser:
    """Recursive parser for Gemma 4's unquoted-key argument syntax."""

    def __init__(self, text: str):
        self.text = text
        self.i = 0

    def parse(self) -> Any:
        value = self._value()
        self._ws()
        if self.i != len(self.text):
            raise ValueError(f"unexpected trailing input at {self.i}")
        return value

    def _ws(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def _value(self) -> Any:
        self._ws()
        if self.text.startswith(_ESCAPE, self.i):
            return self._string()
        if self.i >= len(self.text):
            raise ValueError("missing value")
        char = self.text[self.i]
        if char == "{":
            return self._object()
        if char == "[":
            return self._array()
        return self._literal()

    def _string(self) -> str:
        self.i += len(_ESCAPE)
        end = self.text.find(_ESCAPE, self.i)
        if end == -1:
            raise ValueError("unclosed string delimiter")
        value = self.text[self.i : end]
        self.i = end + len(_ESCAPE)
        return value

    def _key(self) -> str:
        self._ws()
        if self.text.startswith(_ESCAPE, self.i):
            return self._string()
        start = self.i
        while self.i < len(self.text) and self.text[self.i] not in ":,{}[]":
            self.i += 1
        key = self.text[start : self.i].strip()
        if not key:
            raise ValueError("missing object key")
        return key

    def _object(self) -> dict[str, Any]:
        self.i += 1
        result: dict[str, Any] = {}
        self._ws()
        if self.i < len(self.text) and self.text[self.i] == "}":
            self.i += 1
            return result
        while True:
            key = self._key()
            self._ws()
            if self.i >= len(self.text) or self.text[self.i] != ":":
                raise ValueError("missing ':' after object key")
            self.i += 1
            result[key] = self._value()
            self._ws()
            if self.i >= len(self.text):
                raise ValueError("unclosed object")
            char = self.text[self.i]
            self.i += 1
            if char == "}":
                return result
            if char != ",":
                raise ValueError("expected ',' or '}'")

    def _array(self) -> list[Any]:
        self.i += 1
        result: list[Any] = []
        self._ws()
        if self.i < len(self.text) and self.text[self.i] == "]":
            self.i += 1
            return result
        while True:
            result.append(self._value())
            self._ws()
            if self.i >= len(self.text):
                raise ValueError("unclosed array")
            char = self.text[self.i]
            self.i += 1
            if char == "]":
                return result
            if char != ",":
                raise ValueError("expected ',' or ']'")

    def _literal(self) -> Any:
        start = self.i
        while self.i < len(self.text) and self.text[self.i] not in ",}]":
            self.i += 1
        raw = self.text[start : self.i].strip()
        if raw == "true":
            return True
        if raw == "false":
            return False
        if raw == "null":
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if raw:
                return raw
            raise ValueError("missing literal") from None


class Gemma4Renderer:
    """Deterministic renderer for the canonical Gemma 4 instruction models."""

    _config_cls = Gemma4RendererConfig

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: Gemma4RendererConfig | None = None,
        *,
        processor: Any = None,
    ):
        self._tokenizer = tokenizer
        self._processor = processor
        self.config = config or Gemma4RendererConfig()
        model_name = getattr(tokenizer, "name_or_path", "")
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        # The dense/MoE server checkpoints use the July template revision,
        # which pre-closes an empty thought channel when thinking is off.
        # E2B/E4B use the otherwise-identical earlier revision without that
        # prefill. The model-name set is the stable signal. The raw-template
        # probe is only a compatibility fallback for renamed checkpoints and
        # is intentionally narrow; template whitespace/reformatting may defeat it.
        self._prefill_empty_thought = (
            model_name in _EMPTY_THOUGHT_PREFILL_MODELS
            or "<|channel>thought\\n<channel|>" in chat_template
        )
        if self.config.preserve_thinking:
            default_retention = "all"
        elif not self.config.enable_thinking:
            default_retention = "all"
        else:
            default_retention = "tool_cycle"
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config, default_retention
        )

        self._bos = self._required_id("<bos>")
        self._eos = self._required_id("<eos>")
        self._turn_start = self._required_id("<|turn>")
        self._turn_end = self._required_id("<turn|>")
        self._channel_start = self._required_id("<|channel>")
        self._channel_end = self._required_id("<channel|>")
        self._think = self._required_id("<|think|>")
        self._tool_start = self._required_id("<|tool>")
        self._tool_end = self._required_id("<tool|>")
        self._tool_call_start = self._required_id("<|tool_call>")
        self._tool_call_end = self._required_id("<tool_call|>")
        self._tool_response_start = self._required_id("<|tool_response>")
        self._tool_response_end = self._required_id("<tool_response|>")
        self._boi = self._required_id("<|image>")
        self._image = self._required_id("<|image|>")
        self._eoi = self._required_id("<image|>")
        self._model_prefix = self._encode("model\n")
        self._thought_prefix = self._encode("thought\n")
        self._image_cache: dict[str, tuple[Any, int]] = {}

    def _required_id(self, token: str) -> int:
        token_id = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(token_id, int) and token_id != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in Gemma 4 tokenizer vocabulary"
        )
        return token_id

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    @property
    def mm_token_type_id_map(self) -> dict[int, int]:
        return {self._image: 1}

    def _get_processor(self):
        if self._processor is not None:
            return self._processor
        name = getattr(self._tokenizer, "name_or_path", None)
        if not name:
            raise RuntimeError(
                "Gemma4Renderer needs Gemma4Processor for image inputs. Pass "
                "processor=AutoProcessor.from_pretrained(...) or use a tokenizer "
                "with a known name_or_path."
            )
        transformers = _require_transformers("Auto-loading a Gemma 4 processor")
        try:
            self._processor = transformers.AutoProcessor.from_pretrained(name)
        except (ImportError, ValueError) as exc:
            raise RuntimeError(
                "Gemma 4 image rendering requires a Transformers release with "
                "Gemma4Processor support. Upgrade Transformers and retry."
            ) from exc
        return self._processor

    def _process_image(self, part: dict[str, Any]):
        pil = _load_pil_image(part)
        image_hash = _image_hash(pil)
        cached = self._image_cache.get(image_hash)
        if cached is not None:
            output, count = cached
            return output, count, image_hash
        processor = self._get_processor()
        output = processor.image_processor(images=[pil], return_tensors="np")
        counts = output.get("num_soft_tokens_per_image")
        if counts is None:
            raise RuntimeError(
                "Gemma4ImageProcessor did not return num_soft_tokens_per_image."
            )
        count = int(counts[0])
        if len(self._image_cache) >= self.config.image_cache_max:
            self._image_cache.pop(next(iter(self._image_cache)))
        self._image_cache[image_hash] = (output, count)
        return output, count, image_hash

    @classmethod
    def _is_user_query_message(cls, msg: Message) -> bool:
        return msg.get("role") == "user"

    def _emit_image(
        self,
        em: _Emitter,
        part: dict[str, Any],
        mm_hashes: dict[str, list[str]],
        mm_placeholders: dict[str, list[PlaceholderRange]],
        mm_items: dict[str, list[dict[str, Any]]],
        *,
        assistant_body: bool,
    ) -> None:
        output, count, image_hash = self._process_image(part)
        em.special(
            self._boi,
            is_sampled=assistant_body,
            is_content=assistant_body,
        )
        offset = em.cursor()
        for _ in range(count):
            em.special(self._image, is_sampled=assistant_body, is_content=True)
        em.special(
            self._eoi,
            is_sampled=assistant_body,
            is_content=assistant_body,
        )
        mm_hashes.setdefault("image", []).append(image_hash)
        mm_placeholders.setdefault("image", []).append(
            PlaceholderRange(offset=offset, length=count)
        )
        mm_items.setdefault("image", []).append(
            {
                "pixel_values": output["pixel_values"],
                "image_position_ids": output["image_position_ids"],
            }
        )

    def _emit_content(
        self,
        em: _Emitter,
        content: Any,
        role: str,
        mm_hashes: dict[str, list[str]],
        mm_placeholders: dict[str, list[PlaceholderRange]],
        mm_items: dict[str, list[dict[str, Any]]],
    ) -> bool:
        is_assistant = role == "assistant"
        if isinstance(content, str):
            text = _strip_thinking(content) if is_assistant else content.strip()
            em.text(text, is_sampled=is_assistant, is_content=True)
            return bool(text.strip())
        if content is None:
            return False
        if not isinstance(content, list):
            text = str(content).strip()
            em.text(text, is_sampled=is_assistant, is_content=True)
            return bool(text)

        has_content = False
        for item in content:
            if not isinstance(item, Mapping):
                raise ValueError(f"Unexpected Gemma 4 content item: {item!r}")
            # Classify media before text: HF Arrow schema unification
            # (``Dataset.from_list`` over a heterogeneous content list) adds
            # ``text: None`` to every image part, so a key-presence check on
            # ``text`` would swallow images and skip token expansion.
            part = dict(item)
            if _is_image_part(part):
                self._emit_image(
                    em,
                    part,
                    mm_hashes,
                    mm_placeholders,
                    mm_items,
                    assistant_body=is_assistant,
                )
                has_content = True
            elif _is_video_part(part) or part.get("type") in (
                "audio",
                "input_audio",
            ):
                raise NotImplementedError(
                    "Gemma4Renderer currently supports image inputs; audio and "
                    "video inputs are not yet implemented."
                )
            elif part.get("type") == "text" or "text" in part:
                raw = str(part.get("text") or "")
                text = _strip_thinking(raw) if is_assistant else raw.strip()
                em.text(text, is_sampled=is_assistant, is_content=True)
                has_content = has_content or bool(text.strip())
        return has_content

    def _emit_tool_response_body(
        self,
        em: _Emitter,
        name: str,
        response: Any,
        source_idx: int,
        mm_hashes: dict[str, list[str]],
        mm_placeholders: dict[str, list[PlaceholderRange]],
        mm_items: dict[str, list[dict[str, Any]]],
        *,
        emit_start: bool,
        assistant_body: bool = False,
    ) -> None:
        em.set_msg_idx(source_idx)
        if emit_start:
            em.special(
                self._tool_response_start,
                is_sampled=assistant_body,
                is_content=assistant_body,
            )

        media: list[dict[str, Any]] = []
        if isinstance(response, list):
            text = ""
            for raw_part in response:
                if not isinstance(raw_part, Mapping):
                    continue
                # Media before text, for the same schema-unification reason
                # as ``_emit_content``.
                part = dict(raw_part)
                if _is_image_part(part):
                    media.append(part)
                elif _is_video_part(part) or part.get("type") in (
                    "audio",
                    "input_audio",
                ):
                    raise NotImplementedError(
                        "Gemma4Renderer tool responses currently support images only."
                    )
                elif part.get("type") == "text" or "text" in part:
                    text += str(part.get("text") or "")
            response = text

        if isinstance(response, Mapping):
            em.text(
                f"response:{name}",
                is_sampled=assistant_body,
                is_content=assistant_body,
            )
            em.text(
                _format_argument(response, escape_keys=False),
                is_sampled=assistant_body,
                is_content=True,
            )
        else:
            em.text(
                f"response:{name}{{value:",
                is_sampled=assistant_body,
                is_content=assistant_body,
            )
            em.text(
                _format_argument(response, escape_keys=False),
                is_sampled=assistant_body,
                is_content=True,
            )
            em.text(
                "}",
                is_sampled=assistant_body,
                is_content=assistant_body,
            )
        em.special(
            self._tool_response_end,
            is_sampled=assistant_body,
            is_content=assistant_body,
        )
        for part in media:
            self._emit_image(
                em,
                part,
                mm_hashes,
                mm_placeholders,
                mm_items,
                assistant_body=assistant_body,
            )

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        em = _Emitter(self._encode, tokenizer=self._tokenizer)
        mm_hashes: dict[str, list[str]] = {}
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        mm_items: dict[str, list[dict[str, Any]]] = {}

        em.set_msg_idx(-1)
        em.special(self._bos, is_sampled=False, is_content=False)

        first_role = messages[0].get("role")
        first_is_system = first_role in ("system", "developer")
        loop_start = 1 if first_is_system else 0
        previous_message_type: str | None = None

        if self.config.enable_thinking or tools or first_is_system:
            system_idx = 0 if first_is_system else -1
            em.set_msg_idx(system_idx)
            em.special(self._turn_start, is_sampled=False, is_content=False)
            em.text("system\n", is_sampled=False, is_content=False)
            if self.config.enable_thinking:
                em.special(self._think, is_sampled=False, is_content=False)
                em.text("\n", is_sampled=False, is_content=False)
                previous_message_type = "think"
            if first_is_system:
                content = messages[0].get("content")
                if isinstance(content, str):
                    em.text(content.strip(), is_sampled=False, is_content=True)
                elif isinstance(content, list):
                    for item in content:
                        # ``"text" in item`` alone would accept a schema-unified
                        # image part (which carries ``text: None``) and silently
                        # drop the image, so reject media parts explicitly.
                        part = dict(item) if isinstance(item, Mapping) else {}
                        if (
                            not part
                            or "text" not in part
                            or _is_image_part(part)
                            or _is_video_part(part)
                        ):
                            raise ValueError(
                                "Gemma 4 system content lists may contain text parts only."
                            )
                        em.text(
                            str(part.get("text") or "").strip() + " ",
                            is_sampled=False,
                            is_content=True,
                        )
            if tools:
                for tool in tools:
                    if tool.get("type") not in (None, "function"):
                        continue
                    em.special(self._tool_start, is_sampled=False, is_content=False)
                    em.text(
                        _format_function_declaration(tool),
                        is_sampled=False,
                        is_content=False,
                    )
                    em.special(self._tool_end, is_sampled=False, is_content=False)
                previous_message_type = "tool"
            em.special(self._turn_end, is_sampled=False, is_content=False)
            em.text("\n", is_sampled=False, is_content=False)

        loop_indices = list(range(loop_start, len(messages)))
        last_user_pos = -1
        for pos, msg_idx in enumerate(loop_indices):
            if messages[msg_idx].get("role") == "user":
                last_user_pos = pos

        previous_non_tool_role: str | None = None
        consumed_tool_indices: set[int] = set()
        for pos, msg_idx in enumerate(loop_indices):
            msg = messages[msg_idx]
            role = msg.get("role") or ""
            if role == "tool":
                if msg_idx not in consumed_tool_indices:
                    raise ValueError(
                        f"Unconsumed tool message at index {msg_idx}; Gemma 4 tool "
                        "messages must immediately follow an assistant message with "
                        "matching tool_calls."
                    )
                continue

            previous_message_type = None
            wire_role = "model" if role == "assistant" else role
            continues_same_model_turn = (
                wire_role == "model" and previous_non_tool_role == "assistant"
            )
            em.set_msg_idx(msg_idx)
            if not continues_same_model_turn:
                em.special(self._turn_start, is_sampled=False, is_content=False)
                em.text(f"{wire_role}\n", is_sampled=False, is_content=False)

            is_assistant = role == "assistant"
            thinking = msg.get("reasoning") or msg.get("reasoning_content")
            thinking_gate = pos > last_user_pos or (
                self.config.preserve_thinking and bool(msg.get("tool_calls"))
            )
            reemit_disabled_thinking_prefill = (
                is_assistant
                and not continues_same_model_turn
                and not self.config.enable_thinking
                and self._prefill_empty_thought
                and not thinking
            )
            if reemit_disabled_thinking_prefill:
                # The 26B/31B generation prompt already contains this empty
                # thought channel. It is part of the byte prefix under which
                # the completion was sampled, so preserve it in later renders.
                em.special(
                    self._channel_start,
                    is_sampled=is_assistant,
                    is_content=is_assistant,
                )
                em.text(
                    "thought\n",
                    is_sampled=is_assistant,
                    is_content=is_assistant,
                )
                em.special(
                    self._channel_end,
                    is_sampled=is_assistant,
                    is_content=is_assistant,
                )
            elif thinking and thinking_gate:
                em.special(
                    self._channel_start,
                    is_sampled=is_assistant,
                    is_content=is_assistant,
                )
                em.text(
                    f"thought\n{thinking}\n",
                    is_sampled=is_assistant,
                    is_content=True,
                )
                em.special(
                    self._channel_end,
                    is_sampled=is_assistant,
                    is_content=is_assistant,
                )

            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                for tool_call in tool_calls:
                    function = tool_call.get("function") or tool_call
                    name = function.get("name") or ""
                    arguments = _coerce_tool_arguments(function.get("arguments"))
                    body = (
                        "call:" + name + _format_argument(arguments, escape_keys=False)
                    )
                    em.special(
                        self._tool_call_start,
                        is_sampled=is_assistant,
                        is_content=is_assistant,
                    )
                    em.text(
                        body,
                        is_sampled=is_assistant,
                        is_content=is_assistant,
                    )
                    em.special(
                        self._tool_call_end,
                        is_sampled=is_assistant,
                        is_content=is_assistant,
                    )
                previous_message_type = "tool_call"

            rendered_tool_response = False
            legacy_responses = msg.get("tool_responses") or []
            if legacy_responses:
                for response in legacy_responses:
                    self._emit_tool_response_body(
                        em,
                        str(response.get("name") or "unknown"),
                        response.get("response"),
                        msg_idx,
                        mm_hashes,
                        mm_placeholders,
                        mm_items,
                        emit_start=True,
                        assistant_body=is_assistant,
                    )
                    rendered_tool_response = True
                    previous_message_type = "tool_response"
            elif tool_calls:
                scan = msg_idx + 1
                first_response = True
                while scan < len(messages) and messages[scan].get("role") == "tool":
                    response_msg = messages[scan]
                    consumed_tool_indices.add(scan)
                    name = str(response_msg.get("name") or "unknown")
                    for tool_call in tool_calls:
                        if tool_call.get("id") == response_msg.get("tool_call_id"):
                            function = tool_call.get("function") or tool_call
                            name = str(function.get("name") or "unknown")

                    if first_response:
                        em.set_msg_idx(msg_idx)
                        em.special(
                            self._tool_response_start,
                            is_sampled=is_assistant,
                            is_content=is_assistant,
                        )
                    self._emit_tool_response_body(
                        em,
                        name,
                        response_msg.get("content"),
                        scan,
                        mm_hashes,
                        mm_placeholders,
                        mm_items,
                        emit_start=not first_response,
                    )
                    first_response = False
                    rendered_tool_response = True
                    previous_message_type = "tool_response"
                    scan += 1

            em.set_msg_idx(msg_idx)
            has_content = self._emit_content(
                em,
                msg.get("content"),
                role,
                mm_hashes,
                mm_placeholders,
                mm_items,
            )

            next_non_tool_role = None
            for next_idx in loop_indices[pos + 1 :]:
                candidate_role = messages[next_idx].get("role")
                if candidate_role != "tool":
                    next_non_tool_role = candidate_role
                    break
            continues_into_next = (
                wire_role == "model"
                and next_non_tool_role == "assistant"
                and (not tool_calls or rendered_tool_response)
            )

            if previous_message_type == "tool_call" and not rendered_tool_response:
                em.special(
                    self._tool_response_start,
                    is_sampled=is_assistant,
                    is_content=is_assistant,
                )
            elif continues_into_next:
                pass
            elif not (
                rendered_tool_response
                and not has_content
                and next_non_tool_role is None
            ):
                em.special(
                    self._turn_end,
                    is_sampled=is_assistant,
                    is_content=is_assistant,
                )
                em.text("\n", is_sampled=False, is_content=False)

            previous_non_tool_role = role

        if add_generation_prompt:
            em.set_msg_idx(-1)
            if previous_message_type not in ("tool_response", "tool_call"):
                em.special(self._turn_start, is_sampled=False, is_content=False)
                em.text("model\n", is_sampled=False, is_content=False)
                if not self.config.enable_thinking and self._prefill_empty_thought:
                    em.special(
                        self._channel_start,
                        is_sampled=False,
                        is_content=False,
                    )
                    em.text("thought\n", is_sampled=False, is_content=False)
                    em.special(
                        self._channel_end,
                        is_sampled=False,
                        is_content=False,
                    )
            elif (
                previous_message_type == "tool_response" and self.config.enable_thinking
            ):
                em.special(
                    self._channel_start,
                    is_sampled=False,
                    is_content=False,
                )
                em.text("thought\n", is_sampled=False, is_content=False)

        em.finalize()
        multi_modal_data = None
        if mm_hashes or mm_placeholders or mm_items:
            multi_modal_data = MultiModalData(
                mm_hashes=mm_hashes,
                mm_placeholders=mm_placeholders,
                mm_items=mm_items,
            )
        return RenderedTokens(
            token_ids=em.token_ids,
            message_indices=em.message_indices,
            sampled_mask=em.sampled,
            is_content=_content_mask_or_empty(self._tokenizer, em.is_content),
            message_roles=[m.get("role") or "" for m in messages],
            message_tool_names=extract_message_tool_names(messages),
            multi_modal_data=multi_modal_data,
        )

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        ).token_ids

    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def _parse_tool_call(
        self,
        raw: str,
        span: tuple[int, int],
        tools: list[ToolSpec] | None,
    ) -> ParsedToolCall:
        if not raw.startswith("call:") or "{" not in raw or not raw.endswith("}"):
            return ParsedToolCall(
                raw=raw,
                token_span=span,
                status=ToolCallParseStatus.MALFORMED_STRUCTURE,
            )
        head, _, argument_body = raw[5:].partition("{")
        name = head.strip()
        try:
            arguments = _ArgumentParser("{" + argument_body).parse()
        except ValueError:
            return ParsedToolCall(
                raw=raw,
                name=name or None,
                token_span=span,
                status=ToolCallParseStatus.INVALID_JSON,
            )
        if not name:
            return ParsedToolCall(
                raw=raw,
                arguments=arguments,
                token_span=span,
                status=ToolCallParseStatus.MISSING_NAME,
            )
        declared = None
        if tools:
            declared = {
                str(_unwrap_tool(tool).get("name"))
                for tool in tools
                if _unwrap_tool(tool).get("name")
            }
        status = (
            ToolCallParseStatus.UNKNOWN_TOOL
            if declared is not None and name not in declared
            else ToolCallParseStatus.OK
        )
        return ParsedToolCall(
            raw=raw,
            name=name,
            arguments=arguments,
            token_span=span,
            status=status,
        )

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ParsedResponse:
        """Parse a Gemma 4 completion without access to its prompt context.

        After a tool response with thinking enabled, the prompt already emits
        ``<|channel>thought\n`` and the completion contains only the matching
        ``<channel|>`` closer. A lone closer is therefore treated as post-tool
        reasoning. This is necessarily heuristic: without the prompt, a
        malformed first-turn completion with a stray closer is ambiguous.
        """
        stop_ids = {self._turn_end, self._tool_response_start, self._eos}
        end = len(token_ids)
        for i, token_id in enumerate(token_ids):
            if token_id in stop_ids:
                end = i
                break
        ids = list(token_ids[:end])

        prefix = [self._turn_start, *self._model_prefix]
        base_offset = 0
        if ids[: len(prefix)] == prefix:
            ids = ids[len(prefix) :]
            base_offset = len(prefix)

        reasoning: str | None = None
        content_ids: list[int] = []
        cursor = 0
        if ids and ids[0] == self._channel_start:
            channel_end = next(
                (i for i in range(1, len(ids)) if ids[i] == self._channel_end),
                -1,
            )
            thought_start = 1
            if ids[thought_start : thought_start + len(self._thought_prefix)] == (
                self._thought_prefix
            ):
                thought_start += len(self._thought_prefix)
            if channel_end == -1:
                reasoning = self._decode(ids[thought_start:]).strip()
                return ParsedResponse(
                    content="", reasoning_content=reasoning, tool_calls=[]
                )
            reasoning = self._decode(ids[thought_start:channel_end]).strip()
            cursor = channel_end + 1
        elif self.config.enable_thinking:
            # After a tool response, the canonical generation prompt already
            # ends with ``<|channel>thought\n``. The sampled completion therefore
            # starts with the thought body and contains only the closing
            # ``<channel|>`` marker. A lone closer distinguishes that continuation
            # from a normal thinking completion, which samples its own opener.
            channel_end = next(
                (i for i, token_id in enumerate(ids) if token_id == self._channel_end),
                -1,
            )
            if channel_end != -1:
                reasoning = self._decode(ids[:channel_end]).strip()
                cursor = channel_end + 1

        tool_calls: list[ParsedToolCall] = []
        while cursor < len(ids):
            try:
                start = ids.index(self._tool_call_start, cursor)
            except ValueError:
                content_ids.extend(ids[cursor:])
                break
            content_ids.extend(ids[cursor:start])
            try:
                call_end = ids.index(self._tool_call_end, start + 1)
            except ValueError:
                raw = self._decode(ids[start + 1 :]).strip()
                tool_calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=(base_offset + start, base_offset + len(ids)),
                        status=ToolCallParseStatus.UNCLOSED_BLOCK,
                    )
                )
                break
            raw = self._decode(ids[start + 1 : call_end]).strip()
            tool_calls.append(
                self._parse_tool_call(
                    raw,
                    (base_offset + start, base_offset + call_end + 1),
                    tools,
                )
            )
            cursor = call_end + 1

        return ParsedResponse(
            content=self._decode(content_ids).strip(),
            reasoning_content=reasoning,
            tool_calls=tool_calls,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._turn_end, self._tool_response_start, self._eos]

    @staticmethod
    def _merge_multi_modal_data(
        previous: MultiModalData | None,
        hashes: dict[str, list[str]],
        placeholders: dict[str, list[PlaceholderRange]],
        items: dict[str, list[dict[str, Any]]],
    ) -> MultiModalData | None:
        merged_hashes = (
            {key: list(value) for key, value in previous.mm_hashes.items()}
            if previous
            else {}
        )
        merged_placeholders = (
            {key: list(value) for key, value in previous.mm_placeholders.items()}
            if previous
            else {}
        )
        merged_items = (
            {key: list(value) for key, value in previous.mm_items.items()}
            if previous
            else {}
        )
        for key, value in hashes.items():
            merged_hashes.setdefault(key, []).extend(value)
        for key, value in placeholders.items():
            merged_placeholders.setdefault(key, []).extend(value)
        for key, value in items.items():
            merged_items.setdefault(key, []).extend(value)
        if not (merged_hashes or merged_placeholders or merged_items):
            return None
        return MultiModalData(
            mm_hashes=merged_hashes,
            mm_placeholders=merged_placeholders,
            mm_items=merged_items,
        )

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        previous_multi_modal_data: MultiModalData | None = None,
    ) -> RenderedTokens | None:
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None
        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention,
            new_messages,
            is_user_query=self._is_user_query_message,
        ):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._turn_end, self._tool_response_start, self._eos},
            synthesize_close=self._turn_end,
        )
        if previous_ids is None or previous_ids[-1] == self._eos:
            return None

        em = _Emitter(self._encode, tokenizer=self._tokenizer)
        em.token_ids = list(previous_ids)
        em.message_indices = [-1] * len(previous_ids)
        em.sampled = [False] * len(previous_ids)
        em.is_content = [False] * len(previous_ids)
        hashes: dict[str, list[str]] = {}
        placeholders: dict[str, list[PlaceholderRange]] = {}
        items: dict[str, list[dict[str, Any]]] = {}

        if previous_ids[-1] == self._tool_response_start:
            if any(message.get("role") != "tool" for message in new_messages):
                return None
            parsed_calls = [
                call
                for call in self.parse_response(
                    previous_completion_ids, tools=tools
                ).tool_calls
                if call.name
            ]
            if not parsed_calls and any(not m.get("name") for m in new_messages):
                return None
            for i, message in enumerate(new_messages):
                name = message.get("name")
                if not name:
                    if len(parsed_calls) == len(new_messages):
                        name = parsed_calls[i].name
                    elif len(parsed_calls) == 1:
                        name = parsed_calls[0].name
                    else:
                        return None
                self._emit_tool_response_body(
                    em,
                    str(name),
                    message.get("content"),
                    i,
                    hashes,
                    placeholders,
                    items,
                    emit_start=i > 0,
                )
            em.set_msg_idx(-1)
            if self.config.enable_thinking:
                em.special(
                    self._channel_start,
                    is_sampled=False,
                    is_content=False,
                )
                em.text("thought\n", is_sampled=False, is_content=False)
        else:
            em.set_msg_idx(-1)
            em.text("\n", is_sampled=False, is_content=False)
            for i, message in enumerate(new_messages):
                role = message.get("role") or ""
                if role not in ("user", "system", "developer"):
                    return None
                em.set_msg_idx(i)
                em.special(self._turn_start, is_sampled=False, is_content=False)
                em.text(f"{role}\n", is_sampled=False, is_content=False)
                self._emit_content(
                    em,
                    message.get("content"),
                    role,
                    hashes,
                    placeholders,
                    items,
                )
                em.special(self._turn_end, is_sampled=False, is_content=False)
                em.text("\n", is_sampled=False, is_content=False)
            em.set_msg_idx(-1)
            em.special(self._turn_start, is_sampled=False, is_content=False)
            em.text("model\n", is_sampled=False, is_content=False)
            if not self.config.enable_thinking and self._prefill_empty_thought:
                em.special(
                    self._channel_start,
                    is_sampled=False,
                    is_content=False,
                )
                em.text("thought\n", is_sampled=False, is_content=False)
                em.special(
                    self._channel_end,
                    is_sampled=False,
                    is_content=False,
                )

        em.finalize()
        return RenderedTokens(
            token_ids=em.token_ids,
            message_indices=em.message_indices,
            sampled_mask=em.sampled,
            is_content=_content_mask_or_empty(self._tokenizer, em.is_content),
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
            multi_modal_data=self._merge_multi_modal_data(
                previous_multi_modal_data, hashes, placeholders, items
            ),
        )


__all__ = ["Gemma4Renderer"]
