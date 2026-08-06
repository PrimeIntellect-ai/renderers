"""Immutable Hugging Face revisions used by network-backed tests.

Production tokenizer loading intentionally follows the requested model unless
the repository executes trusted remote code. Tests have a different contract:
parity expectations must not change because an upstream model owner moves
``main``. Every real model asset loaded by the test suite therefore goes
through :func:`load_test_tokenizer` with a full commit SHA.

For canonical Meta Llama IDs the revision belongs to the audited ``unsloth``
tokenizer mirror selected by ``TOKENIZER_SOURCE_OVERRIDES``. Kimi revisions
must match ``TRUSTED_REVISIONS`` because those tokenizers execute repository
code and cannot be overridden by a test pin.
"""

from __future__ import annotations

from typing import Any

from renderers.base import TRUSTED_REVISIONS, load_tokenizer


MODEL_REVISIONS: dict[str, str] = {
    "MiniMaxAI/MiniMax-M2.5": "f710177d938eff80b684d42c5aa84b382612f21f",
    "PrimeIntellect/Qwen3-0.6B": "15aba67c8ec68aeac96f57ba7e31d373564ee03c",
    "PrimeIntellect/Qwen3-1.7B": "61a79cd101170642a653d1d44d18f69f88f54e08",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3-VL-4B-Instruct": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c",
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-35B-A3B": "59d61f3ce65a6d9863b86d2e96597125219dc754",
    "Qwen/Qwen3.5-122B-A10B": "dc4d348443bc740c68e2d77492492c11606384d5",
    "Qwen/Qwen3.5-397B-A17B": "8472618112abcbd45acbcdc58436aff4233c23f7",
    "Qwen/Qwen3.6-35B-A3B": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    "THUDM/GLM-4.5-Air": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
    "deepseek-ai/DeepSeek-R1": "56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad",
    "deepseek-ai/DeepSeek-V3": "e815299b0bcbac849fa540c768ef21845365c9eb",
    "meta-llama/Llama-3.2-1B-Instruct": "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
    "meta-llama/Llama-3.2-3B-Instruct": "006f5dcd1393c3add266de40994ba96225e9689d",
    "moonshotai/Kimi-K2-Instruct": TRUSTED_REVISIONS["moonshotai/Kimi-K2-Instruct"],
    "moonshotai/Kimi-K2.5": TRUSTED_REVISIONS["moonshotai/Kimi-K2.5"],
    "moonshotai/Kimi-K2.6": TRUSTED_REVISIONS["moonshotai/Kimi-K2.6"],
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "2d59de1cbd51c0adf384eb906b766d1aee0e0517",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "d51eab0d1f979ebc26b546e634a04f450d99158e",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "624ba927cfbef0427354998700de3d51173c8c04",
    "openai/gpt-oss-20b": "6cee5e81ee83917806bbde320786a8fb61efebee",
    "poolside/Laguna-XS-2.1": "e9df9a59996d790b94b70f3fef343fe1d9e34bdf",
    "poolside/Laguna-XS.2": "69e3f4046616e40fb55ac54e0e2e6accbe5cadfe",
    "tencent/Hy3": "a960ebc3da325ba167f069f76c41eb62c9280d22",
    "zai-org/GLM-4.7-Flash": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
    "zai-org/GLM-5": "4e6698ba8e85059d749020e3c4d2123719f23926",
    "zai-org/GLM-5.1": "26e1bd6e011feb778d25ae34b09b07074139d92d",
}


def model_revision(model_name: str) -> str:
    """Return the immutable test revision for ``model_name``."""
    try:
        return MODEL_REVISIONS[model_name]
    except KeyError as exc:
        raise KeyError(
            f"No immutable test revision registered for {model_name!r}. "
            "Resolve and review a full Hugging Face commit SHA before adding "
            "this model to a network-backed test."
        ) from exc


def load_test_tokenizer(model_name: str):
    """Load a real tokenizer at the suite's reviewed immutable revision."""
    return load_tokenizer(model_name, revision=model_revision(model_name))


def processor_load_kwargs(model_name: str) -> dict[str, Any]:
    """Pinned, security-aware kwargs for ``AutoProcessor.from_pretrained``."""
    return {
        "revision": model_revision(model_name),
        "trust_remote_code": model_name in TRUSTED_REVISIONS,
    }
