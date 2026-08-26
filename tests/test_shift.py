"""Unit tests for the SHIFT vector math, TLR@k, and baselines (no downloads)."""

import pytest
import torch

from shift.apply import apply_language_shift, get_doc_language
from shift.baselines import apply_centering, apply_lir, compute_language_means, compute_language_subspaces
from shift.langvec import compute_language_vectors
from shift.metrics import compute_tlr_at_k


def test_language_vectors_are_mean_offsets():
    torch.manual_seed(0)
    z_en = torch.randn(10, 4)
    offset = torch.tensor([1.0, -2.0, 0.5, 0.0])
    z_de = z_en + offset

    vectors = compute_language_vectors({"en": z_en, "de": z_de}, source_lang="en")
    assert set(vectors) == {"de"}
    assert torch.allclose(vectors["de"], offset, atol=1e-6)


def test_apply_shift_only_moves_target_language_docs():
    embs = torch.ones(3, 4)
    vectors = {"de": torch.tensor([1.0, 1.0, 1.0, 1.0])}
    shifted = apply_language_shift(embs, ["en", "de", "unknown"], vectors, alpha=0.5, source_lang="en")

    assert torch.allclose(shifted[0], embs[0])          # source language untouched
    assert torch.allclose(shifted[1], embs[1] - 0.5)    # target shifted by alpha * v
    assert torch.allclose(shifted[2], embs[2])          # unknown untouched
    assert torch.allclose(embs, torch.ones(3, 4))       # input not mutated


def test_shift_recovers_source_space_exactly():
    """If every doc is exactly source + v_l, alpha=1 recovers the source docs."""
    torch.manual_seed(1)
    src = torch.randn(5, 8)
    v = {"zh": torch.randn(8)}
    docs = src + v["zh"]
    shifted = apply_language_shift(docs, ["zh"] * 5, v, alpha=1.0, source_lang="en")
    assert torch.allclose(shifted, src, atol=1e-6)


def test_get_doc_language():
    assert get_doc_language("doc_de_12", {}) == "de"
    assert get_doc_language("en_0_3", {}) == "en"
    assert get_doc_language("anything", {"lang": "JA"}) == "ja"
    assert get_doc_language("xyz", {}) == "unknown"


def test_tlr_at_k():
    qrels = {"q1": {"d_en": 1, "d_de": 1, "d_fr": 1}}
    doc_langs = {"d_en": "en", "d_de": "de", "d_fr": "fr", "d_x": "es"}
    # Top-2 contains d_de but not d_fr -> TLR@2 = 1/2
    results = {"q1": {"d_en": 0.9, "d_de": 0.8, "d_fr": 0.1, "d_x": 0.5}}
    tlr, n = compute_tlr_at_k(qrels, results, doc_langs, k=2, query_lang="en")
    assert n == 1
    assert tlr == pytest.approx(0.5)

    # A query whose only relevant doc is in the query language is excluded
    qrels2 = {"q1": {"d_en": 1}}
    tlr2, n2 = compute_tlr_at_k(qrels2, results, doc_langs, k=2, query_lang="en")
    assert tlr2 is None and n2 == 0


def test_centering_subtracts_language_centroid():
    torch.manual_seed(2)
    emb_de = torch.randn(20, 6) + 5.0
    means = compute_language_means({"de": emb_de})
    centered = apply_centering(emb_de, ["de"] * 20, means)
    assert torch.allclose(centered.mean(dim=0), torch.zeros(6), atol=1e-5)


def test_lir_removes_top_singular_direction():
    torch.manual_seed(3)
    base = torch.randn(50, 6)
    direction = torch.nn.functional.normalize(torch.randn(6), dim=0)
    emb = base + 20.0 * torch.randn(50, 1) * direction  # dominant language direction

    subspaces = compute_language_subspaces({"xx": emb}, num_components=3)
    cleaned = apply_lir(emb, ["xx"] * 50, subspaces, num_remove=1)

    # After removal, the embeddings have no component along the top direction
    v1 = subspaces["xx"][:, 0]
    assert torch.allclose(cleaned @ v1, torch.zeros(50), atol=1e-4)
    # And the top direction found by SVD is (anti)parallel to the injected one
    assert abs(torch.dot(v1, direction).item()) > 0.99
