"""Regenerate the checked-in renderer behavior corpus.

Run from the repository root:

    uv run python -m tests.generate_renderer_goldens
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.golden_corpus import GOLDEN_CASES, build_golden_case


OUTPUT_PATH = Path(__file__).with_name("golden_renderer_outputs.json")


def main() -> None:
    cases = {}
    for case in GOLDEN_CASES:
        print(f"rendering {case.slug} ({case.model_name})", flush=True)
        cases[case.slug] = build_golden_case(case)
    payload = {"schema_version": 1, "cases": cases}
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
