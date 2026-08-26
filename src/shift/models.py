"""Model loading and per-model prompt/prefix registry.

Each retriever family requires its own query/document prefixes and loading
options (dtype, trust_remote_code). This module centralizes the rules used
across all experiments (see Appendix B, Table 5 of the paper).
"""

from __future__ import annotations

import os

import torch
from sentence_transformers import SentenceTransformer

QWEN_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)

# Matched by substring of the (lowercased) model name; first match wins.
_QUERY_PREFIX_RULES: list[tuple[str, str]] = [
    ("qwen", QWEN_QUERY_PROMPT),  # Qwen3-Embedding and gte-Qwen2
    ("e5", "query: "),
    ("snowflake-arctic-embed", "query: "),
    ("jina", "Represent the query for retrieving evidence documents: "),
    ("embeddinggemma", "task: search result | query: "),
    ("llama-nemotron-embed", "query: "),
    ("mxbai", "Represent this sentence for searching relevant passages: "),
    ("nomic", "search_query: "),
]

_DOC_PREFIX_RULES: list[tuple[str, str]] = [
    ("e5", "passage: "),
    ("jina", "Represent the document for retrieval: "),
    ("embeddinggemma", "title: none | text: "),
    ("llama-nemotron-embed", "passage: "),
    ("nomic", "search_document: "),
]

_TRUST_REMOTE_CODE_KEYS = ("gte", "jina", "llama-nemotron-embed", "nomic", "snowflake-arctic-embed-m")
_BF16_KEYS = ("gte", "jina", "llama-nemotron-embed", "qwen", "granite")
_FP16_KEYS = ("mxbai-embed",)


def _match_prefix(model_name: str, rules: list[tuple[str, str]]) -> str:
    lower = model_name.lower()
    for key, prefix in rules:
        if key in lower:
            return prefix
    return ""


def query_prefix(model_name: str) -> str:
    """Query prompt/prefix for a retriever ("" if none)."""
    return _match_prefix(model_name, _QUERY_PREFIX_RULES)


def doc_prefix(model_name: str) -> str:
    """Document prompt/prefix for a retriever ("" if none)."""
    return _match_prefix(model_name, _DOC_PREFIX_RULES)


def add_query_prefix(texts: list[str], model_name: str) -> list[str]:
    prefix = query_prefix(model_name)
    return [prefix + t for t in texts] if prefix else list(texts)


def add_doc_prefix(texts: list[str], model_name: str) -> list[str]:
    prefix = doc_prefix(model_name)
    return [prefix + t for t in texts] if prefix else list(texts)


def load_st_model(model_name: str, max_seq_length: int = 512) -> SentenceTransformer:
    """Load a SentenceTransformer with the per-family options used in the paper."""
    lower = model_name.lower()

    kwargs: dict = {}
    if any(k in lower for k in _TRUST_REMOTE_CODE_KEYS):
        kwargs["trust_remote_code"] = True
    if any(k in lower for k in _BF16_KEYS):
        kwargs["model_kwargs"] = {"dtype": torch.bfloat16}
    elif any(k in lower for k in _FP16_KEYS):
        kwargs["model_kwargs"] = {"dtype": torch.float16}

    model = SentenceTransformer(model_name, **kwargs)
    model.max_seq_length = max_seq_length

    # Decoder-based retrievers: disable the generation KV cache
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "use_cache"):
            module.config.use_cache = False

    return model


def get_safe_model_name(model_name: str) -> str:
    """Filesystem-safe name for a model id or local path."""
    name = model_name.rstrip("/")
    if os.path.exists(name):
        name = os.path.basename(name)
    return name.replace("/", "_").replace(" ", "_")


def encode_devices() -> str | list[str] | None:
    """Device spec for SentenceTransformer.encode (multi-GPU aware)."""
    if not torch.cuda.is_available():
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    num_gpus = (
        len([x for x in visible.split(",") if x.strip()]) if visible is not None
        else torch.cuda.device_count()
    )
    if num_gpus <= 1:
        return "cuda:0"
    return [f"cuda:{i}" for i in range(num_gpus)]
