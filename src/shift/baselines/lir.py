"""Language Information Removal (LIR) baseline (Yang et al., 2021; Appendix C.1).

For each language l, stack embeddings of texts written in l into X_l and take
the top-r right-singular vectors V_{l,r} from its SVD as the principal
language-identity directions. Each embedding is updated by removing its
projection onto that subspace:

    x -> x - V_{l,r} V_{l,r}^T x

Following Yang et al. (2021), r = 1 works best and is the default. Applied to
both documents (at indexing) and queries (at retrieval).
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_language_subspaces(
    embeddings_by_lang: dict[str, Tensor],
    num_components: int = 20,
) -> dict[str, Tensor]:
    """Per-language top-k right-singular vectors: {lang: (d, k) tensor}.

    num_components caps how many components are stored; how many are removed
    is decided at application time (r <= num_components).
    """
    subspaces = {}
    for lang, emb in embeddings_by_lang.items():
        # torch.linalg.svd: X = U S Vh, rows of Vh are right-singular vectors
        _, _, vh = torch.linalg.svd(emb.float(), full_matrices=False)
        subspaces[lang] = vh[:num_components].T.contiguous().cpu()  # (d, k)
    return subspaces


def apply_lir(
    embeddings: Tensor,
    langs: list[str],
    subspaces: dict[str, Tensor],
    num_remove: int = 1,
) -> Tensor:
    """Remove each row's language subspace projection.

    Unknown languages (no subspace) are unchanged.
    """
    if num_remove <= 0:
        return embeddings.clone()

    result = embeddings.clone()
    for lang, subspace in subspaces.items():
        rows = [i for i, l in enumerate(langs) if l == lang]
        if not rows:
            continue
        v = subspace[:, :num_remove].to(device=result.device, dtype=result.dtype)  # (d, r)
        x = result[rows]  # (n, d)
        result[rows] = x - (x @ v) @ v.T
    return result
