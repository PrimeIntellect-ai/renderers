"""Immutable test assets; Llama revisions refer to the audited tokenizer mirrors."""

from pathlib import Path

from renderers.base import load_tokenizer

MODEL_REVISIONS = {
    "MiniMaxAI/MiniMax-M2.5": "2515a3ea0afec1dd22d0c7794b3a91557804d195",
    "PrimeIntellect/Qwen3-0.6B": "15aba67c8ec68aeac96f57ba7e31d373564ee03c",
    "PrimeIntellect/Qwen3-1.7B": "61a79cd101170642a653d1d44d18f69f88f54e08",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c",
    "Qwen/Qwen3-VL-4B-Instruct": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-122B-A10B": "dc4d348443bc740c68e2d77492492c11606384d5",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-35B-A3B": "59d61f3ce65a6d9863b86d2e96597125219dc754",
    "Qwen/Qwen3.5-397B-A17B": "8472618112abcbd45acbcdc58436aff4233c23f7",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.6-35B-A3B": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "Qwen/Qwen3.8-Flash-Next": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
    "THUDM/GLM-4.5-Air": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
    "deepseek-ai/DeepSeek-R1": "56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad",
    "deepseek-ai/DeepSeek-V3": "e815299b0bcbac849fa540c768ef21845365c9eb",
    "deepseek-ai/DeepSeek-V4-Flash-0731": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    "google/gemma-4-26B-A4B-it": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    "google/gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "google/gemma-4-E2B-it": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "google/gemma-4-E4B-it": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "meta-llama/Llama-3.2-1B-Instruct": "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
    "meta-llama/Llama-3.2-3B-Instruct": "006f5dcd1393c3add266de40994ba96225e9689d",
    "meta-models/Muse-Glimmer-30B": "cec94fcfd6512f2cfe708138c32533b346ea4f69",
    "moonshotai/Kimi-K2-Instruct": "fd1984e2b7a3350dbf7305fe73a4ede25c14de50",
    "moonshotai/Kimi-K2.5": "4d01dfe0332d63057c186e0b262165819efb6611",
    "moonshotai/Kimi-K2.6": "2755962d07cb42aa2d988a35bcb65cd4a9c2de82",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "bf77c3174f68ad409e1c2aa60daeb46e32d1c606",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "2dc98e2afe4face0e4ce40972a915c45368bd34a",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "77df655d5e9f8362164ed14dd8b48f8bce657498",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": "a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
    "openai/gpt-oss-120b": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    "openai/gpt-oss-20b": "6cee5e81ee83917806bbde320786a8fb61efebee",
    "poolside/Laguna-M.1": "2bf8a4ab6d2e8e1e0d92a41318532a0fee1640d6",
    "poolside/Laguna-S-2.1": "0f573140834b11cfac0c2af97a101a7a69a13e22",
    "poolside/Laguna-XS-2.1": "c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db",
    "poolside/Laguna-XS.2": "a397bde04501591d0ac3d74a5d9d0a4ec298ffb8",
    "tencent/Hy3": "a960ebc3da325ba167f069f76c41eb62c9280d22",
    "thinkingmachines/Inkling": "828496eeae4c243ff1a22f7f28ff83694f2f7bc9",
    "thinkingmachines/Inkling-Small": "8cc5877b44d343f88b92086aa1fb72897950f06a",
    "zai-org/GLM-4.7-Flash": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
    "zai-org/GLM-5": "c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2",
    "zai-org/GLM-5.1": "26e1bd6e011feb778d25ae34b09b07074139d92d",
    "zai-org/GLM-5.3": "ba32f8c19c8738e2c57c71798c601d477a7bf844",
    "zai-org/GLM-5.3-BF16": "abf42e6c0e53bc29a875511301e7b7fcd220315b",
    "zai-org/GLM-5.3-Flash-BF16": "f9fed3bbc82291073373d67452e9f25317ef7577",
}


def model_revision(model: str) -> str:
    # The optional private VLM fixture can only resolve an existing local snapshot.
    if model == "nvidia/NVIDIA-Nemotron-3.5-Super-EA-09112026":
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model, "tokenizer_config.json")
        if isinstance(path, str):
            return Path(path).parent.name
    try:
        return MODEL_REVISIONS[model]
    except KeyError as exc:
        raise KeyError(f"No immutable test revision registered for {model!r}") from exc


def load_test_tokenizer(model: str):
    return load_tokenizer(model, revision=model_revision(model))


def load_test_processor(model: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model,
        revision=model_revision(model),
        trust_remote_code=model
        in {
            "moonshotai/Kimi-K2.5",
            "moonshotai/Kimi-K2.6",
            "nvidia/NVIDIA-Nemotron-3.5-Super-EA-09112026",
        },
    )
