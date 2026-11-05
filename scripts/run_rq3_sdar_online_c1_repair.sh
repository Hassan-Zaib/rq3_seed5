#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/tiny-imagenet-200}"
RUN_ROOT="${RUN_ROOT:-results_sdar_online_repair_c1}"
FIXED_PROBE_DIR="${FIXED_PROBE_DIR:-fixed_probe}"
SDAR_STEPS="${SDAR_STEPS:-40000}"
SDAR_BATCH_SIZE="${SDAR_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ATTACK_POOL="${ATTACK_POOL:-train}"

echo "===== RQ3 pilot repair: ConvNeXt-Tiny + online SDAR none C1 only ====="
echo "RUN_ROOT=${RUN_ROOT}"
echo "SDAR_STEPS=${SDAR_STEPS}"
echo "ATTACK_POOL=${ATTACK_POOL}"

python scripts/verify_rq3_implementation.py

if [[ ! -f "$FIXED_PROBE_DIR/fixed_probe_indices.npy" ]]; then
  python scripts/make_fixed_probe.py \
    --data-root "$DATA_ROOT" \
    --out-dir "$FIXED_PROBE_DIR" \
    --num-images 20 \
    --seed 9090
else
  echo "Reusing immutable fixed probe from $FIXED_PROBE_DIR"
fi

python scripts/check_shapes.py --cut c1

python scripts/run_sdar_on_convnext_smashed.py \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --fixed-probe-dir "$FIXED_PROBE_DIR" \
  --cut c1 \
  --defense none \
  --steps "$SDAR_STEPS" \
  --batch-size "$SDAR_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --attack-pool "$ATTACK_POOL" \
  --fg-optimizer adamw \
  --fg-lr 0.002 \
  --fg-weight-decay 0.05 \
  --fg-schedule warmup_cosine \
  --fg-warmup-steps 3000 \
  --fg-min-lr 0.000001 \
  --fg-label-smoothing 0.1 \
  --fg-grad-clip 1.0

python scripts/print_rq3_sdar_online_results.py --run-root "$RUN_ROOT" --defense none
echo "[RQ3 online SDAR C1 repair pilot finished]"
