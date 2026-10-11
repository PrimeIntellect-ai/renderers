"""Pin assets used by unchanged upstream fixtures in an isolated HF cache.

Fixtures request `main`. Populate that ref with reviewed snapshots, then run
upstream offline. This changes neither fixture code nor its expected results.
"""

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(__file__).parent
pins = json.loads((root / "upstream_tokenizers.json").read_text())
native = json.loads((root / "tokenizers.json").read_text())
for model in (
    "Qwen/Qwen3.5-397B-A17B",
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2.5",
    "google/gemma-4-E2B-it",
):
    pins[model] = native[model]["revision"]
cache = Path(os.environ["HF_HUB_CACHE"])
for model, revision in pins.items():
    path = Path(
        snapshot_download(
            model,
            revision=revision,
            cache_dir=cache,
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.py",
                "*.model",
                "*.txt",
                "*.tiktoken",
            ],
        )
    )
    refs = path.parent.parent / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(revision)
    print(f"{model}@{revision}", flush=True)
