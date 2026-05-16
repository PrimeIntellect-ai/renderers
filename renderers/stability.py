from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Boundary = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class RenderStability:
    """Declared append-boundaries that preserve an existing rendered prefix."""

    preserves_through: frozenset[Boundary]


FULLY_STABLE = RenderStability(frozenset({"user", "assistant", "tool"}))
STABLE_IN_TOOL_CYCLE = RenderStability(frozenset({"tool"}))
OPAQUE = RenderStability(frozenset())
