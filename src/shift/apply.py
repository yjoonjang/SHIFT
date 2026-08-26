"""Index-side language shift (Algorithm 1 of the paper).

Each document embedding pre-labeled with language l is calibrated during
indexing as

    z_tilde = z - alpha * V_l        (l != l_src)

Source-language documents (and documents with unknown language) are left
unchanged. Queries are never transformed.
"""

from __future__ import annotations

from torch import Tensor

from shift import ALL_LANGS


def get_doc_language(doc_id: str, doc: dict, supported: list[str] | None = None) -> str:
    """Infer a document's language from its metadata or ID pattern.

    Prefers the corpus's explicit "lang" field; falls back to id patterns
    like "en_..." / "doc_en_..." / "..._en".
    """
    supported = supported or ALL_LANGS

    if "lang" in doc:
        lang = str(doc["lang"]).lower()
        if lang in supported or lang == "en":
            return lang

    if isinstance(doc_id, str):
        lower = doc_id.lower()
        parts = lower.split("_")
        for lang in supported + ["en"]:
            if lower.startswith(f"{lang}_") or lower.endswith(f"_{lang}"):
                return lang
            if len(parts) >= 3 and parts[0] == "doc" and parts[1] == lang:
                return lang
        if lower.startswith("cn_"):
            return "zh"

    return "unknown"


def apply_language_shift(
    embeddings: Tensor,
    doc_langs: list[str],
    lang_vectors: dict[str, Tensor],
    alpha: float,
    source_lang: str = "en",
) -> Tensor:
    """Apply the index-side shift to a corpus embedding matrix.

    Args:
        embeddings: (N, d) corpus embeddings (unnormalized).
        doc_langs: Language label per row.
        lang_vectors: {lang: (d,) tensor} relative language vectors.
        alpha: Scale factor in [0, 1].
        source_lang: Documents in this language are left unchanged.

    Returns:
        A new (N, d) tensor with the shift applied.
    """
    if alpha == 0.0 or not lang_vectors:
        return embeddings.clone()

    shifted = embeddings.clone()
    for lang, vector in lang_vectors.items():
        if lang == source_lang:
            continue
        rows = [i for i, dl in enumerate(doc_langs) if dl == lang]
        if not rows:
            continue
        v = vector.to(device=shifted.device, dtype=shifted.dtype)
        shifted[rows] -= alpha * v
    return shifted
