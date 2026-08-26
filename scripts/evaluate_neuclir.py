"""NeuCLIR 2023 MLIR evaluation — Base vs SHIFT (Appendix F).

English queries retrieve from a pooled Chinese + Russian corpus
(NeuCLIR2023RetrievalHardNegatives via mteb, with the official English queries
and cross-lingual qrels from neuclir/bench). Documents are shifted by their
own language vector; Persian is excluded (outside the vector-estimation
corpus).

Requires the "neuclir" extra:  uv sync --extra neuclir

Usage:
    python scripts/evaluate_neuclir.py --model intfloat/multilingual-e5-large --alpha 0.6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict

import mteb
from datasets import load_dataset

from shift.apply import apply_language_shift
from shift.langvec import lang_vector_path, load_language_vectors
from shift.models import add_doc_prefix, add_query_prefix, encode_devices, get_safe_model_name, load_st_model

import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_setting, rank_results  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ISO_TO_LANG = {"rus": "ru", "zho": "zh"}
TASK_NAME = "NeuCLIR2023RetrievalHardNegatives"


def build_neuclir_data():
    """Build (queries, corpus_texts, doc_langs, qrels) for the pooled ru+zh track."""
    tasks = mteb.get_tasks(tasks=[TASK_NAME], languages=list(ISO_TO_LANG.keys()))
    task = tasks[0]
    task.load_data()

    corpus: dict[str, str] = {}
    doc_langs: dict[str, str] = {}
    track_query_ids: set[str] = set()

    for iso, lang in ISO_TO_LANG.items():
        if iso not in task.dataset:
            continue
        d = task.dataset[iso]["test"]
        for row in d["corpus"]:
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            content = (title + "\n" + text).strip() if title else text
            doc_id = f"{iso}::{row['id']}"
            corpus[doc_id] = content
            doc_langs[doc_id] = lang
        for row in d["queries"]:
            track_query_ids.add(str(row["id"]))

    eng = load_dataset("neuclir/bench", "queries")["eng"]
    eng_queries = {str(r["id"]): r["query"] for r in eng}
    queries = {qid: eng_queries[qid] for qid in track_query_ids if qid in eng_queries}

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    qr = load_dataset("neuclir/bench", "qrels")
    for iso in ISO_TO_LANG:
        for row in qr[iso]:
            qid = str(row["id"])
            if qid not in queries:
                continue
            doc_id = f"{row.get('ignore')}::{row['docid']}"
            if doc_id in corpus and int(row["relevance"]) > 0:
                qrels[qid][doc_id] = 1

    queries = {q: t for q, t in queries.items() if q in qrels}
    return queries, corpus, doc_langs, dict(qrels)


def main():
    parser = argparse.ArgumentParser(description="NeuCLIR 2023 Base vs SHIFT")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--alpha", "-a", type=float, default=1.0)
    parser.add_argument("--lang_vectors_dir", type=str, default="lang_vectors")
    parser.add_argument("--results_dir", type=str, default="results_neuclir")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tlr_k", type=int, default=20)
    args = parser.parse_args()

    model = load_st_model(args.model)
    devices = encode_devices()

    vec_path = lang_vector_path(args.lang_vectors_dir, args.model, "en")
    lang_vectors = load_language_vectors(vec_path) if os.path.exists(vec_path) else None
    if lang_vectors is None:
        logger.warning("Language vectors not found at %s — only the base setting will run", vec_path)

    queries, corpus, doc_langs, qrels = build_neuclir_data()
    logger.info("Queries: %d, Corpus: %d", len(queries), len(corpus))

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())

    query_texts = add_query_prefix([queries[q] for q in query_ids], args.model)
    corpus_texts = add_doc_prefix([corpus[c] for c in corpus_ids], args.model)

    q_embs = model.encode(query_texts, batch_size=args.batch_size, convert_to_tensor=True,
                          show_progress_bar=True, device=devices).cpu().float()
    c_embs = model.encode(corpus_texts, batch_size=args.batch_size, convert_to_tensor=True,
                          show_progress_bar=True, device=devices).cpu().float()

    doc_lang_list = [doc_langs[c] for c in corpus_ids]
    settings = [("base", 0.0)]
    if lang_vectors is not None:
        settings.append((f"shift_alpha{args.alpha:g}", args.alpha))

    output = {"model": args.model, "benchmark": TASK_NAME, "settings": {}}
    for setting_name, alpha in settings:
        shifted = apply_language_shift(c_embs, doc_lang_list, lang_vectors or {}, alpha, source_lang="en")
        results = rank_results(q_embs, shifted, query_ids, corpus_ids)
        metrics = evaluate_setting(qrels, results, doc_langs, args.tlr_k, "en")
        metrics["alpha"] = alpha
        output["settings"][setting_name] = metrics
        logger.info("[%s] nDCG@20=%.4f", setting_name, metrics["ndcg"]["NDCG@20"])

    out_root = os.path.join(args.results_dir, get_safe_model_name(args.model))
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f"{TASK_NAME}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
