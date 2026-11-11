#!/usr/bin/env bash
# Reproduce every number in the paper.
#
#   bash run_all.sh            # full suite (11 benchmarks), CPU, ~25 min
#   bash run_all.sh quick      # 3 benchmarks, ~2 min, wiring check only
#
# Each step writes one directory under results/, mirroring the layout of the
# released runs. See README.md for the mapping from directories to paper items.
#
# Environment:
#   CF_DEVICE   torch device for training/audit (default: cpu)
#   CF_PYTHON   interpreter to use            (default: python)
#   CF_EXT_ROOT directory holding genius.mat / Penn94.mat for the large-scale
#               run (step 8); the step is skipped when unset
set -euo pipefail
cd "$(dirname "$0")"

DS_FULL="cora,citeseer,pubmed,tolokers,amazon_ratings,chameleon_filtered,actor,squirrel_filtered,wisconsin,cornell,texas"
DS_QUICK="cora,texas,actor"
if [ "${1:-full}" = "quick" ]; then DS="$DS_QUICK"; SUF="_quick"; else DS="$DS_FULL"; SUF=""; fi
DEV="${CF_DEVICE:-cpu}"
PY="${CF_PYTHON:-python}"

run_main() { # <outdir> <extra CF_* env>
  echo "== $1 =="
  env CF_CONFLICT_MODE=curate2 $2 $PY src/run_main.py \
      --outdir "results/$1${SUF}" --device "$DEV" --datasets "$DS" \
      --edge-mode sup --edge-space auto --edge-feat full
}

# [1] Main method: soft conflict + symmetric aggregation + neighbour background.
#     Produces Tables I-II, the audit verdict, and the per-graph figures.
run_main main ""

# [2-5] Component ablation (Table III, first block): one operator added per row.
run_main ablation/binary_deg          "CF_CONFLICT_MODE=hard"
run_main ablation/soft_deg            "CF_AGG=mean CF_COMBINE=none"
run_main ablation/soft_sym            "CF_AGG=sym CF_COMBINE=none"
run_main ablation/soft_deg_background "CF_AGG=mean CF_COMBINE=avg"

# [6] Label-free edge scorer (Table III): cosine-threshold scorer, no labels.
echo "== ablation: label-free edge scorer =="
env CF_CONFLICT_MODE=curate2 $PY src/run_main.py \
    --outdir "results/ablation/label_free_scorer${SUF}" --device "$DEV" --datasets "$DS" \
    --edge-mode unsup --edge-space auto --edge-feat full

# [7] Validation-tuned candidate selection (Table III).
echo "== ablation: validation-tuned candidate selection =="
env CF_CONFLICT_MODE=nbr6 $PY src/run_main.py \
    --outdir "results/ablation/val_candidate_selection${SUF}" --device "$DEV" --datasets "$DS" \
    --edge-mode sup --edge-space auto --edge-feat full

echo "== ablation: synthetic sweep + GNN depth =="
$PY src/run_ablation.py --which all --outdir "results/ablation${SUF}"

# [8] Large-scale graphs (Section Scalability, "Two million-edge graphs").
#      Requires genius.mat and Penn94.mat in CF_EXT_ROOT (see README).
if [ -n "${CF_EXT_ROOT:-}" ]; then
  echo "== large-scale: genius + Penn94 =="
  env CF_CONFLICT_MODE=curate2 CF_EXT_ROOT="$CF_EXT_ROOT" $PY src/run_main.py \
      --outdir "results/large_scale${SUF}" --device "$DEV" \
      --datasets genius,penn94 --edge-mode sup --edge-space auto --edge-feat full
else
  echo "== large-scale skipped (set CF_EXT_ROOT to enable) =="
fi

# Optional: restack the 1x4 diagnostic figures into the paper's 2x2 panels.
# for ds in cora squirrel_filtered; do
#   $PY src/make_figures.py "results/main/figures/${ds}_conflict.png" "${ds}_conflict_2x2.png"
# done

echo
echo "done -> results/main, results/ablation*${SUF}"
