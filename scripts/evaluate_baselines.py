"""Evaluate the training-free post-hoc baselines (Appendix C): Centering and LIR.

Both estimate per-language statistics from the same mMARCO pairs used by SHIFT
and, unlike SHIFT, transform BOTH document embeddings (by document language)
and query embeddings (by query language):

    centering  x -> x - mu_l                    (language centroid)
    lir        x -> x - V_{l,r} V_{l,r}^T x     (top-r SVD language directions, r=1)

Statistics are cached per model under --stats_dir.

Usage:
    python scripts/evaluate_baselines.py --model BAAI/bge-m3 --method centering \
        --pairs_data data/mmarco_pairs
    python scripts/evaluate_baselines.py --model BAAI/bge-m3 --method lir \
        --pairs_data data/mmarco_pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch
from datasets import load_from_disk

from shift import ALL_LANGS
from shift.apply import get_doc_language
from shift.baselines import apply_centering, apply_lir, compute_language_means, compute_language_subspaces
from shift.data import ALL_BENCHMARKS, DEFAULT_DATA_DIR, load_benchmark, resolve_benchmark_dir
from shift.models import add_doc_prefix, add_query_prefix, encode_devices, get_safe_model_name, load_st_model

import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_setting, rank_results  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_or_load_stats(args, model, devices) -> dict[str, torch.Tensor]:
    safe_name = get_safe_model_name(args.model)
    stats_path = os.path.join(args.stats_dir, f"{safe_name}_{args.method}.pt")
    if os.path.exists(stats_path):
        logger.info("Loading cached %s stats from %s", args.method, stats_path)
        return torch.load(stats_path, weights_only=False)

    if args.pairs_data is None:
        raise ValueError(f"No cached stats at {stats_path}; pass --pairs_data to compute them")

    dataset = load_from_disk(args.pairs_data)
    langs = [l for l in ALL_LANGS if f"positive_{l}" in dataset.column_names]

    embeddings_by_lang = {}
    for lang in langs:
        texts = add_doc_prefix(dataset[f"positive_{lang}"], args.model)
        logger.info("Encoding %s passages (%d)...", lang, len(texts))
        emb = model.encode(
            texts, batch_size=args.batch_size, show_progress_bar=True,
            convert_to_numpy=True, device=devices, normalize_embeddings=False,
        )
        embeddings_by_lang[lang] = torch.from_numpy(emb)

    if args.method == "centering":
        stats = compute_language_means(embeddings_by_lang)
    else:
        stats = compute_language_subspaces(embeddings_by_lang)

    os.makedirs(args.stats_dir, exist_ok=True)
    torch.save(stats, stats_path)
    logger.info("Saved %s stats to %s", args.method, stats_path)
    return stats


def transform(embs: torch.Tensor, langs: list[str], stats, method: str, num_remove: int) -> torch.Tensor:
    if method == "centering":
        return apply_centering(embs, langs, stats)
    return apply_lir(embs, langs, stats, num_remove=num_remove)


def main():
    parser = argparse.ArgumentParser(description="Centering / LIR baseline evaluation")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["centering", "lir"])
    parser.add_argument("--num_remove", type=int, default=1, help="LIR: SVD components to remove")
    parser.add_argument("--query_lang", type=str, default="en")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=ALL_BENCHMARKS)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pairs_data", type=str, default=None,
                        help="Aligned mMARCO table (needed on the first run per model)")
    parser.add_argument("--stats_dir", type=str, default="baseline_stats")
    parser.add_argument("--results_dir", type=str, default="results_baselines")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tlr_k", type=int, default=20)
    args = parser.parse_args()

    model = load_st_model(args.model)
    devices = encode_devices()
    stats = compute_or_load_stats(args, model, devices)

    safe_name = get_safe_model_name(args.model)
    out_root = os.path.join(args.results_dir, safe_name, args.method)
    os.makedirs(out_root, exist_ok=True)

    for benchmark_name in args.benchmarks:
        benchmark_dir = resolve_benchmark_dir(
            benchmark_name, query_lang=args.query_lang, data_dir=args.data_dir
        )
        logger.info("=== %s (%s) ===", benchmark_name, args.method)
        queries, corpus, qrels = load_benchmark(benchmark_dir)

        query_ids = list(queries.keys())
        corpus_ids = list(corpus.keys())
        doc_langs = {cid: get_doc_language(cid, corpus[cid]) for cid in corpus_ids}

        query_texts = add_query_prefix([queries[q] for q in query_ids], args.model)
        corpus_texts = add_doc_prefix([corpus[c]["text"].strip() for c in corpus_ids], args.model)

        q_embs = model.encode(
            query_texts, batch_size=args.batch_size, convert_to_tensor=True,
            show_progress_bar=True, device=devices,
        ).cpu().float()
        c_embs = model.encode(
            corpus_texts, batch_size=args.batch_size, convert_to_tensor=True,
            show_progress_bar=True, device=devices,
        ).cpu().float()

        q_langs = [args.query_lang] * len(query_ids)
        doc_lang_list = [doc_langs[c] for c in corpus_ids]

        q_trans = transform(q_embs, q_langs, stats, args.method, args.num_remove)
        c_trans = transform(c_embs, doc_lang_list, stats, args.method, args.num_remove)

        results = rank_results(q_trans, c_trans, query_ids, corpus_ids)
        metrics = evaluate_setting(qrels, results, doc_langs, args.tlr_k, args.query_lang)
        metrics["method"] = args.method
        if args.method == "lir":
            metrics["num_remove"] = args.num_remove

        tlr = metrics[f"tlr@{args.tlr_k}"]
        logger.info(
            "[%s] nDCG@20=%.4f Recall@20=%.4f TLR@%d=%s",
            args.method, metrics["ndcg"]["NDCG@20"], metrics["recall"]["Recall@20"],
            args.tlr_k, f"{tlr:.4f}" if tlr is not None else "n/a",
        )

        out_path = os.path.join(out_root, f"{benchmark_name}_{args.query_lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "benchmark": benchmark_name, **metrics}, f,
                      indent=2, ensure_ascii=False)
        logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
