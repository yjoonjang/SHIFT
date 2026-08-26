"""Estimate relative language vectors from aligned mMARCO passage pairs (Eq. 1).

Reads the aligned parallel table produced by
``scripts/prepare_data/prepare_mmarco_pairs.py`` (columns ``positive_{lang}``),
encodes the passages of the source language and of every target language with
the retriever (unnormalized, document prefix applied), and saves the mean
offsets as ``{model}_lang_vectors_src-{source_lang}.pt``.

Usage:
    python scripts/compute_lang_vectors.py --model intfloat/multilingual-e5-large \
        --pairs_data data/mmarco_pairs --out_dir lang_vectors

    # Non-English source language (Section 5.3)
    python scripts/compute_lang_vectors.py --model BAAI/bge-m3 \
        --pairs_data data/mmarco_pairs --out_dir lang_vectors --source_lang zh
"""

from __future__ import annotations

import argparse
import logging

import torch
from datasets import load_from_disk

from shift import ALL_LANGS
from shift.langvec import compute_language_vectors, lang_vector_path, save_language_vectors
from shift.models import add_doc_prefix, encode_devices, load_st_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Estimate relative language vectors")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--pairs_data", type=str, required=True,
                        help="Aligned mMARCO parallel table (prepare_mmarco_pairs.py output)")
    parser.add_argument("--out_dir", type=str, default="lang_vectors")
    parser.add_argument("--source_lang", type=str, default="en")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    dataset = load_from_disk(args.pairs_data)
    langs = [l for l in ALL_LANGS if f"positive_{l}" in dataset.column_names]
    if args.source_lang not in langs:
        raise ValueError(f"source_lang={args.source_lang} not found in the pairs data ({langs})")
    logger.info("Pairs: %d rows, languages: %s", len(dataset), langs)

    model = load_st_model(args.model)
    devices = encode_devices()

    embeddings_by_lang: dict[str, torch.Tensor] = {}
    for lang in langs:
        texts = add_doc_prefix(dataset[f"positive_{lang}"], args.model)
        logger.info("Encoding %s passages (%d)...", lang, len(texts))
        emb = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=devices,
            normalize_embeddings=False,
        )
        embeddings_by_lang[lang] = torch.from_numpy(emb)

    vectors = compute_language_vectors(embeddings_by_lang, source_lang=args.source_lang)
    for lang, v in vectors.items():
        logger.info("||v_%s|| = %.6f", lang, torch.norm(v).item())

    out_path = lang_vector_path(args.out_dir, args.model, args.source_lang)
    save_language_vectors(vectors, args.model, args.source_lang, out_path)
    logger.info("Saved %d language vectors to %s", len(vectors), out_path)


if __name__ == "__main__":
    main()
