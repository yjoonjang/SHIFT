"""Loaders for the BEIR-style MLIR benchmark JSONL files.

Directory layout (one directory per benchmark):
    queries.jsonl: {"_id": ..., "text": ...} or {qid: text}
    corpus.jsonl:  {doc_id: {"text": ..., "lang": ...}}
    qrels.jsonl:   {query_id: {doc_id: relevance}}

Benchmarks are read from a local directory when present, and otherwise
downloaded from the HuggingFace Hub (the exact files used in the paper).
They can also be rebuilt from the original sources with:
    python scripts/prepare_data/build_benchmarks.py --output_dir data/benchmarks
"""

from __future__ import annotations

import json
import os

from huggingface_hub import snapshot_download

DEFAULT_DATA_DIR = "data/benchmarks"
DEFAULT_BENCHMARK_REPO = "yjoonjang/mlir-benchmarks"

ALL_BENCHMARKS = ["Belebele_test", "MLQA_test", "XQuAD_test", "MultiEup_test"]


def resolve_benchmark_dir(
    benchmark_name: str,
    query_lang: str = "en",
    data_dir: str = DEFAULT_DATA_DIR,
    repo_id: str = DEFAULT_BENCHMARK_REPO,
) -> str:
    """Return a local directory containing the benchmark's JSONL files.

    Resolution order: a local root produced by build_benchmarks.py (with an
    ``MLIR-{query_lang}/`` variant subdirectory), a directory containing the
    benchmark folders directly, then the HuggingFace Hub.
    """
    candidates = [
        os.path.join(data_dir, f"MLIR-{query_lang}", benchmark_name),
        os.path.join(data_dir, benchmark_name),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    subdir = f"MLIR-{query_lang}"
    root = snapshot_download(
        repo_id, repo_type="dataset", allow_patterns=[f"{subdir}/{benchmark_name}/*"]
    )
    hub_dir = os.path.join(root, subdir, benchmark_name)
    if os.path.isdir(hub_dir):
        return hub_dir
    raise FileNotFoundError(
        f"Benchmark '{benchmark_name}' (query_lang={query_lang}) not found under {data_dir} "
        f"or in https://huggingface.co/datasets/{repo_id}"
    )


def load_queries(path: str) -> dict[str, str]:
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "_id" in data and "text" in data:
                queries[data["_id"]] = data["text"]
            else:
                queries.update(data)
    return queries


def load_corpus(path: str) -> dict[str, dict]:
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            corpus.update(json.loads(line))
    return corpus


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qrels.update(json.loads(line))
    return qrels


def load_benchmark(benchmark_dir: str) -> tuple[dict, dict, dict]:
    """Load (queries, corpus, qrels) from a benchmark directory."""
    return (
        load_queries(os.path.join(benchmark_dir, "queries.jsonl")),
        load_corpus(os.path.join(benchmark_dir, "corpus.jsonl")),
        load_qrels(os.path.join(benchmark_dir, "qrels.jsonl")),
    )
