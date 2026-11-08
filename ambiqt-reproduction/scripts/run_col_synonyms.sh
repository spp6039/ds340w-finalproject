#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$ROOT/results"

cd "$ROOT/benchmark/col-synonyms"

python evaluate.py \
  -f ../../outputs/col-synonyms/logical-beam.json \
  -p template_filled \
  -k 5 \
  2>&1 | tee "$ROOT/results/col_synonyms_baseline_results.txt"
