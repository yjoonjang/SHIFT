"""Relative language vector estimation (Eq. 1 of the paper).

For a source language l_src and each target language l_tgt, the relative
language vector is the mean embedding offset over N aligned translation pairs:

    V_tgt = (1/N) * sum_i ( f(D_i^tgt) - f(D_i^src) )

Embeddings are used unnormalized; document prefixes are applied per model.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from shift.models import get_safe_model_name


def compute_language_vectors(
    embeddings_by_lang: dict[str, Tensor],
    source_lang: str,
) -> dict[str, Tensor]:
    """Compute relative language vectors from pre-computed embeddings.

    Args:
        embeddings_by_lang: {lang: (N, d) tensor} of aligned document
            embeddings; must include source_lang. Rows must be aligned across
            languages (same document index = same content).
        source_lang: Reference language (embeddings shift toward it).

    Returns:
        {lang: (d,) tensor} for every language except source_lang.
    """
    z_src = embeddings_by_lang[source_lang]
    vectors = {}
    for lang, z_lang in embeddings_by_lang.items():
        if lang == source_lang:
            continue
        if z_lang.shape != z_src.shape:
            raise ValueError(
                f"Embedding shape mismatch for {lang}: {tuple(z_lang.shape)} vs "
                f"source {tuple(z_src.shape)}"
            )
        vectors[lang] = (z_lang - z_src).mean(dim=0).cpu()
    return vectors


def lang_vector_path(out_dir: str, model_name: str, source_lang: str = "en") -> str:
    """Canonical save path for a model's language vectors,
    e.g. ``BAAI_bge-m3_lang_vectors_src-en.pt``."""
    safe_name = get_safe_model_name(model_name)
    return os.path.join(out_dir, f"{safe_name}_lang_vectors_src-{source_lang}.pt")


def save_language_vectors(
    vectors: dict[str, Tensor], model_name: str, source_lang: str, path: str
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model_name": model_name, "source_lang": source_lang}
    for lang, v in vectors.items():
        payload[f"v_{lang}"] = v
    torch.save(payload, path)


def load_language_vectors(path: str) -> dict[str, Tensor]:
    """Load language vectors as {lang: (d,) tensor}."""
    payload = torch.load(path, weights_only=False)
    return {k[len("v_"):]: v for k, v in payload.items() if k.startswith("v_")}
