"""End-to-end test of the evaluation pipeline on a synthetic benchmark.

Runs the actual ranking + BEIR metric + TLR path from scripts/evaluate.py with
random embeddings — no model downloads. Constructs a corpus where each English
query's positives exist in en/de/fr; the "biased" embedding space ranks the
same-language positive highest, and the shift vector corrects it.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from evaluate import evaluate_setting, rank_results  # noqa: E402


def test_rank_and_metrics_pipeline():
    torch.manual_seed(0)
    dim = 16

    # Two queries, each with parallel positives in en/de/fr + one distractor
    query_ids = ["q_a", "q_b"]
    corpus_ids = ["doc_en_0", "doc_de_0", "doc_fr_0", "doc_en_1", "doc_de_1", "doc_fr_1", "doc_en_x"]
    doc_langs = {c: c.split("_")[1] for c in corpus_ids}
    qrels = {
        "q_a": {"doc_en_0": 1, "doc_de_0": 1, "doc_fr_0": 1},
        "q_b": {"doc_en_1": 1, "doc_de_1": 1, "doc_fr_1": 1},
    }

    sem_a, sem_b, sem_x = torch.randn(3, dim)
    v_de, v_fr = torch.randn(2, dim) * 3  # large language offsets

    q_embs = torch.stack([sem_a, sem_b])
    c_embs = torch.stack([
        sem_a, sem_a + v_de, sem_a + v_fr,
        sem_b, sem_b + v_de, sem_b + v_fr,
        sem_x,
    ])

    results = rank_results(q_embs, c_embs, query_ids, corpus_ids)
    metrics = evaluate_setting(qrels, results, doc_langs, k=3, query_lang="en")

    # Sanity: BEIR metrics parsed, TLR present
    assert 0.0 <= metrics["ndcg"]["NDCG@3"] <= 1.0
    assert metrics["tlr_num_queries"] == 2

    # Now apply the exact language shift — every positive collapses onto its
    # query's semantics, so all 3 positives fill the top-3 and TLR@3 becomes 1.
    from shift.apply import apply_language_shift

    shifted = apply_language_shift(
        c_embs, [doc_langs[c] for c in corpus_ids],
        {"de": v_de, "fr": v_fr}, alpha=1.0, source_lang="en",
    )
    results_shift = rank_results(q_embs, shifted, query_ids, corpus_ids)
    metrics_shift = evaluate_setting(qrels, results_shift, doc_langs, k=3, query_lang="en")

    assert metrics_shift["tlr@3"] == 1.0
    assert metrics_shift["ndcg"]["NDCG@3"] >= metrics["ndcg"]["NDCG@3"]
