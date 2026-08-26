"""Language-wise centering baseline (Libovický et al., 2020; Appendix C.1).

Estimates a per-language centroid mu_l as the mean embedding of texts written
in language l, and re-centers each embedding as x -> x - mu_l. Applied to both
documents (at indexing) and queries (at retrieval), each with its own
language's centroid.
"""

from __future__ import annotations

from torch import Tensor


def compute_language_means(embeddings_by_lang: dict[str, Tensor]) -> dict[str, Tensor]:
    """Per-language mean embeddings: {lang: (d,) tensor}."""
    return {lang: emb.mean(dim=0).cpu() for lang, emb in embeddings_by_lang.items()}


def apply_centering(
    embeddings: Tensor,
    langs: list[str],
    means: dict[str, Tensor],
) -> Tensor:
    """Subtract each row's language centroid. Unknown languages are unchanged."""
    centered = embeddings.clone()
    for lang, mean in means.items():
        rows = [i for i, l in enumerate(langs) if l == lang]
        if not rows:
            continue
        mu = mean.to(device=centered.device, dtype=centered.dtype)
        centered[rows] -= mu
    return centered
