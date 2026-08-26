"""Evaluate a retriever on the MLIR benchmarks, Base vs SHIFT.

Encodes queries and corpus once per benchmark, then applies the index-side
language shift for every requested alpha — so an alpha sweep costs no extra
encoding. Reports nDCG@k, Recall@k (BEIR/pytrec_eval), and TLR@k.

Usage:
    # Base + SHIFT at one alpha
    python scripts/evaluate.py --model intfloat/multilingual-e5-large --alphas 0.6

    # Alpha sweep (Figure 3)
    python scripts/evaluate.py --model BAAI/bge-m3 --alphas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

    # Non-English source queries (Section 5.3); expects vectors estimated with
    # the same --source_lang and benchmarks built with that query language
    python scripts/evaluate.py --model BAAI/bge-m3 --alphas 1.0 --query_lang zh
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers.util import cos_sim

from shift.apply import apply_language_shift, get_doc_language
from shift.data import ALL_BENCHMARKS, DEFAULT_DATA_DIR, load_benchmark, resolve_benchmark_dir
from shift.langvec import lang_vector_path, load_language_vectors
from shift.metrics import compute_tlr_at_k
from shift.models import add_doc_prefix, add_query_prefix, encode_devices, get_safe_model_name, load_st_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

K_VALUES = [1, 3, 5, 10, 20, 100, 1000]


def rank_results(q_embs: torch.Tensor, c_embs: torch.Tensor, query_ids, corpus_ids) -> dict:
    """Full-ranking cosine retrieval results: {qid: {doc_id: score}}."""
    scores = cos_sim(q_embs, c_embs)  # normalizes internally
    results = {}
    for i, qid in enumerate(query_ids):
        row = scores[i]
        order = torch.argsort(row, descending=True)
        results[qid] = {corpus_ids[j]: float(row[j]) for j in order.tolist() if corpus_ids[j] != qid}
    return results


def evaluate_setting(qrels, results, doc_langs, k, query_lang) -> dict:
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=K_VALUES)
    tlr, n_tlr_queries = compute_tlr_at_k(qrels, results, doc_langs, k=k, query_lang=query_lang)
    return {
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
        f"tlr@{k}": tlr,
        "tlr_num_queries": n_tlr_queries,
    }


def main():
    parser = argparse.ArgumentParser(description="Base vs SHIFT MLIR evaluation")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--alphas", "-a", type=float, nargs="+", default=[1.0],
                        help="SHIFT scale factors to evaluate (base alpha=0 always included)")
    parser.add_argument("--query_lang", type=str, default="en",
                        help="Source query language of the benchmark variant")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=ALL_BENCHMARKS)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Benchmark root (build_benchmarks.py output)")
    parser.add_argument("--lang_vectors_dir", type=str, default="lang_vectors")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tlr_k", type=int, default=20)
    args = parser.parse_args()

    model = load_st_model(args.model)
    devices = encode_devices()

    vec_path = lang_vector_path(args.lang_vectors_dir, args.model, args.query_lang)
    if os.path.exists(vec_path):
        lang_vectors = load_language_vectors(vec_path)
        logger.info("Loaded language vectors from %s (%d languages)", vec_path, len(lang_vectors))
    else:
        lang_vectors = None
        logger.warning("Language vectors not found at %s — only the base setting will run", vec_path)

    safe_name = get_safe_model_name(args.model)
    out_root = os.path.join(args.results_dir, safe_name)
    os.makedirs(out_root, exist_ok=True)

    for benchmark_name in args.benchmarks:
        benchmark_dir = resolve_benchmark_dir(
            benchmark_name, query_lang=args.query_lang, data_dir=args.data_dir
        )
        logger.info("=== %s (query_lang=%s) ===", benchmark_name, args.query_lang)
        queries, corpus, qrels = load_benchmark(benchmark_dir)
        logger.info("Queries: %d, Corpus: %d", len(queries), len(corpus))

        query_ids = list(queries.keys())
        corpus_ids = list(corpus.keys())
        doc_langs = {cid: get_doc_language(cid, corpus[cid]) for cid in corpus_ids}

        query_texts = add_query_prefix([queries[q] for q in query_ids], args.model)
        corpus_texts = add_doc_prefix(
            [
                (f"{corpus[c].get('title', '')} {corpus[c]['text']}".strip()
                 if corpus[c].get("title") else corpus[c]["text"].strip())
                for c in corpus_ids
            ],
            args.model,
        )

        logger.info("Encoding queries...")
        q_embs = model.encode(
            query_texts, batch_size=args.batch_size, convert_to_tensor=True,
            show_progress_bar=True, device=devices,
        ).cpu().float()
        logger.info("Encoding corpus...")
        c_embs = model.encode(
            corpus_texts, batch_size=args.batch_size, convert_to_tensor=True,
            show_progress_bar=True, device=devices,
        ).cpu().float()

        doc_lang_list = [doc_langs[c] for c in corpus_ids]
        settings = [("base", 0.0)]
        if lang_vectors is not None:
            settings += [(f"shift_alpha{alpha:g}", alpha) for alpha in args.alphas]

        output = {
            "model": args.model,
            "benchmark": benchmark_name,
            "query_lang": args.query_lang,
            "settings": {},
        }
        for setting_name, alpha in settings:
            logger.info("[%s] ranking...", setting_name)
            shifted = apply_language_shift(
                c_embs, doc_lang_list, lang_vectors or {}, alpha, source_lang=args.query_lang
            )
            results = rank_results(q_embs, shifted, query_ids, corpus_ids)
            metrics = evaluate_setting(qrels, results, doc_langs, args.tlr_k, args.query_lang)
            metrics["alpha"] = alpha
            output["settings"][setting_name] = metrics
            tlr = metrics[f"tlr@{args.tlr_k}"]
            logger.info(
                "[%s] nDCG@20=%.4f Recall@20=%.4f TLR@%d=%s",
                setting_name,
                metrics["ndcg"]["NDCG@20"],
                metrics["recall"]["Recall@20"],
                args.tlr_k,
                f"{tlr:.4f}" if tlr is not None else "n/a",
            )

        out_path = os.path.join(out_root, f"{benchmark_name}_{args.query_lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
