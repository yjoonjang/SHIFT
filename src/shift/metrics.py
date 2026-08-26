"""Target-Languages Recall@k (TLR@k), Section 4.3 of the paper.

For each query q with language L_q, restrict the relevant set to documents
whose language differs from L_q:

    TLR@k = (1/|Q|) * sum_q |top_k(q) ∩ R_q^tgt| / |R_q^tgt|

where R_q^tgt = {d in relevant(q) | L(d) != L_q}. Queries without any
target-language relevant document are excluded from the average.
"""

from __future__ import annotations


def compute_tlr_at_k(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    doc_langs: dict[str, str],
    k: int = 20,
    query_lang: str = "en",
) -> tuple[float | None, int]:
    """Compute TLR@k.

    Args:
        qrels: {qid: {doc_id: relevance}}.
        results: {qid: {doc_id: score}} retrieval results (full ranking
            recommended so top-k is exact).
        doc_langs: {doc_id: language}.
        k: Cutoff.
        query_lang: The query (source) language; relevant documents in this
            language are excluded from the target set.

    Returns:
        (tlr_at_k, num_queries_counted); tlr_at_k is None when no query has a
        target-language relevant document.
    """
    per_query = []

    for qid, rels in qrels.items():
        if qid not in results:
            continue

        target_rels = [
            doc_id
            for doc_id, rel in rels.items()
            if rel > 0 and doc_langs.get(doc_id, "unknown") not in (query_lang, "unknown")
        ]
        if not target_rels:
            continue

        ranked = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
        top_docs = {doc_id for doc_id, _ in ranked[:k]}

        hits = sum(1 for d in target_rels if d in top_docs)
        per_query.append(hits / len(target_rels))

    if not per_query:
        return None, 0
    return sum(per_query) / len(per_query), len(per_query)
