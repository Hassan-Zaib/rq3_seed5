#!/usr/bin/env bash
# One RQ3 seed-5 job: online SDAR on ConvNeXt-Tiny / Tiny ImageNet, one defense and one cut.
# The python command is the DGX seed-5 command: dgx/run_seeds45_one_defense.sh runs scripts/run_rq3_sdar_online.sh
# with SEED=5 RUN_SMOKE=0, and that script's defaults are copied here unchanged. Only two paths differ:
# --run-root is results_seed5 and --fixed-probe-dir is this repository's fixed_probe/.
# check_seed5.sh confirms the two commands are the same.
#
# Usage: bash scripts/run_one.sh DEFENSE CUT      e.g. bash scripts/run_one.sh q4 c2
# Output: results_seed5/<defense>/<cut>/sdar/ (full_sdar_summary.json when done), log: logs_seed5/<defense>_<cut>.log
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_SEED=5
if [[ $# -ne 2 ]]; then
  echo "usage: bash scripts/run_one.sh DEFENSE CUT   (seed 5 only)" >&2
  exit 2
fi
if [[ -n "${SEED:-}" && "${SEED}" != "${RUN_SEED}" ]]; then
  echo "this repository runs seed ${RUN_SEED} only (SEED=${SEED} given)" >&2
  exit 2
fi
DEFENSE="$1"
CUT="$2"
case "$DEFENSE" in none|q4|q2|noise0.01|noise0.05) ;; *) echo "unknown defense: $DEFENSE" >&2; exit 2 ;; esac
case "$CUT" in c1|c2|c3|c4) ;; *) echo "unknown cut: $CUT" >&2; exit 2 ;; esac

DATA_ROOT=data/tiny-imagenet-200
RUN_ROOT=results_seed5
FIXED_PROBE_DIR=fixed_probe

CMD=(
  python scripts/run_sdar_on_convnext_smashed.py
  --data-root "$DATA_ROOT"
  --run-root "$RUN_ROOT"
  --fixed-probe-dir "$FIXED_PROBE_DIR"
  --cut "$CUT"
  --seed "$RUN_SEED"
  --split-seed 42
  --defense "$DEFENSE"
  --steps 40000
  --batch-size 256
  --num-workers 0
  --attack-pool train
  --victim-train-transform
  --fg-optimizer adamw
  --fg-lr 0.002
  --fg-weight-decay 0.05
  --fg-schedule warmup_cosine
  --fg-warmup-steps 3000
  --fg-min-lr 0.000001
  --fg-label-smoothing 0.1
  --fg-mix-prob 1.0
  --fg-mixup-alpha 0.8
  --fg-cutmix-alpha 1.0
  --fg-cutmix-switch-prob 0.5
  --fg-grad-clip 1.0
  --amp
  --amp-dtype bf16
  --model-ema
  --model-ema-decay 0.9999
)

summary="${RUN_ROOT}/${DEFENSE}/${CUT}/sdar/full_sdar_summary.json"
if [[ -f "$summary" ]]; then
  echo "Skip completed job ${DEFENSE} ${CUT}: ${summary} exists"
  exit 0
fi

# Used by check_seed5.sh: print the command and stop.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

mkdir -p logs_seed5
exec > >(tee -a "logs_seed5/${DEFENSE}_${CUT}.log") 2>&1
echo "===== RQ3 seed ${RUN_SEED} online SDAR ConvNeXt-Tiny TinyImageNet ${DEFENSE} ${CUT} ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ====="

# ---- checks before training (fail here, not hours later) --------------------------------------------------
for f in wnids.txt train val/val_annotations.txt val/images; do
  if [[ ! -e "${DATA_ROOT}/${f}" ]]; then
    echo "missing ${DATA_ROOT}/${f}: link your Tiny ImageNet as data/tiny-imagenet-200 (README section 2)" >&2
    exit 1
  fi
done
python - "$FIXED_PROBE_DIR" <<'EOF'
import hashlib, sys
from pathlib import Path
expected = {
    "fixed_probe_images.npy": "e0b16d8c278f85b85dd5b7aa18c1ce2cd44f6e822edd21b86f2fe10f4c8b7217",
    "fixed_probe_images_norm.npy": "7e92b7c12f5e0adf04c46ae425bab6b4962cdc21411aeef6013517d1211566a3",
    "fixed_probe_indices.npy": "3816111b0b920b4a3a04058186c19240b86dd9090e34c56a88888e33d81308e2",
    "fixed_probe_labels.npy": "b817dfed4b67dced25ba37baf123ea9b29d967753a64d2e4eba6ef161efa564d",
}
for name, sha in expected.items():
    got = hashlib.sha256((Path(sys.argv[1]) / name).read_bytes()).hexdigest()
    if got != sha:
        sys.exit(f"fixed probe file {name} was changed (sha256 {got})")
    print(f"fixed probe {name} sha256 ok")
EOF
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "PyTorch does not see a GPU" >&2
  exit 1
fi
# LPIPS is only computed after training; load it now so missing AlexNet weights fail at the start.
python -c "import lpips; lpips.LPIPS(net='alex')"
# The same pre-checks the DGX runner does before training.
python scripts/verify_rq3_implementation.py
python scripts/check_shapes.py --cut "$CUT"

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
echo "finished ${DEFENSE} ${CUT}: ${summary}"
