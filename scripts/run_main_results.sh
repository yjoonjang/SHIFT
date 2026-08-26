#!/bin/bash
# Reproduce the main results (Tables 2 & 3): for each retriever, estimate
# language vectors once, then evaluate Base vs SHIFT across the alpha grid on
# all four MLIR benchmarks.
#
# Usage:
#   bash scripts/run_main_results.sh
#
# Environment overrides:
#   PAIRS_DATA (default: data/mmarco_pairs)  ALPHAS  MODELS

set -uo pipefail

PAIRS_DATA="${PAIRS_DATA:-data/mmarco_pairs}"
ALPHAS="${ALPHAS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"

DEFAULT_MODELS=(
    "google/embeddinggemma-300m"
    "intfloat/multilingual-e5-large"
    "BAAI/bge-m3"
    "Qwen/Qwen3-Embedding-0.6B"
    "nvidia/llama-nemotron-embed-1b-v2"
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
)
if [ -n "${MODELS:-}" ]; then
    read -r -a MODEL_LIST <<< "$MODELS"
else
    MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

for MODEL in "${MODEL_LIST[@]}"; do
    echo ""
    echo "=============================================="
    echo "  ${MODEL}"
    echo "=============================================="

    uv run python scripts/compute_lang_vectors.py \
        --model "$MODEL" \
        --pairs_data "$PAIRS_DATA" \
        --out_dir lang_vectors

    # shellcheck disable=SC2086
    uv run python scripts/evaluate.py \
        --model "$MODEL" \
        --alphas $ALPHAS
done

echo "All models done. Results in results/"
